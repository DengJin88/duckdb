#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"

#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/common/types/null_value.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"

using namespace duckdb;
using namespace std;

PhysicalHashAggregate::PhysicalHashAggregate(vector<TypeId> types, vector<unique_ptr<Expression>> expressions,
                                             PhysicalOperatorType type)
    : PhysicalHashAggregate(types, move(expressions), {}, type) {
}

PhysicalHashAggregate::PhysicalHashAggregate(vector<TypeId> types, vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p, PhysicalOperatorType type)
    : PhysicalSink(type, types), groups(move(groups_p)) {
	// get a list of all aggregates to be computed
	// fake a single group with a constant value for aggregation without groups
	if (this->groups.size() == 0) {
		auto ce = make_unique<BoundConstantExpression>(Value::TINYINT(42));
		this->groups.push_back(move(ce));
		is_implicit_aggr = true;
	} else {
		is_implicit_aggr = false;
	}
	for (auto &expr : groups) {
		group_types.push_back(expr->return_type);
	}
	all_combinable = true;
	for (auto &expr : expressions) {
		assert(expr->expression_class == ExpressionClass::BOUND_AGGREGATE);
		assert(expr->IsAggregate());
		auto &aggr = (BoundAggregateExpression &)*expr;
		bindings.push_back(&aggr);

		aggregate_types.push_back(aggr.return_type);
		if (aggr.children.size()) {
			for (idx_t i = 0; i < aggr.children.size(); ++i) {
				payload_types.push_back(aggr.children[i]->return_type);
			}
		} else {
			// COUNT(*)
			payload_types.push_back(TypeId::INT64);
		}
		if (!aggr.function.combine) {
			all_combinable = false;
		}
		aggregates.push_back(move(expr));
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class HashAggregateGlobalState : public GlobalOperatorState {
public:
	HashAggregateGlobalState(vector<TypeId> &group_types, vector<TypeId> &payload_types,
	                         vector<BoundAggregateExpression *> &bindings)
	    : is_empty(true) {
		//  		ht = make_unique<SuperLargeHashTable>(1024, group_types, payload_types, bindings);
	}

	//	//! The lock for updating the global aggregate state
	//	std::mutex lock;
	//	//! The aggregate HT
	//	unique_ptr<SuperLargeHashTable> ht;
	//! Whether or not any tuples were added to the HT
	bool is_empty;
};

struct AggregateObject {
	AggregateObject(AggregateFunction function, idx_t child_count, idx_t payload_size, bool distinct,
	                TypeId return_type)
	    : function(move(function)), child_count(child_count), payload_size(payload_size), distinct(distinct),
	      return_type(return_type) {
	}

	AggregateFunction function;
	idx_t child_count;
	idx_t payload_size;
	bool distinct;
	TypeId return_type;

	static vector<AggregateObject> CreateAggregateObjects(vector<BoundAggregateExpression *> bindings) {
		vector<AggregateObject> aggregates;
		for (auto &binding : bindings) {
			auto payload_size = binding->function.state_size();
			aggregates.push_back(AggregateObject(binding->function, binding->children.size(), payload_size,
			                                     binding->distinct, binding->return_type));
		}
		return aggregates;
	}
};

class HashAggregateLocalState : public LocalSinkState {
public:
	HashAggregateLocalState(vector<unique_ptr<Expression>> &groups, vector<BoundAggregateExpression *> &aggr_bindings,
	                        vector<TypeId> &group_types, vector<TypeId> &payload_types)
	    : group_executor(groups), group_width(0), payload_width(0) {
		for (auto &aggr : aggr_bindings) {
			if (aggr->children.size()) {
				for (idx_t i = 0; i < aggr->children.size(); ++i) {
					payload_executor.AddExpression(*aggr->children[i]);
				}
			}
		}
		group_chunk.Initialize(group_types);
		if (payload_types.size() > 0) {
			payload_chunk.Initialize(payload_types);
		}

		for (idx_t i = 0; i < group_types.size(); i++) {
			group_width += GetTypeIdSize(group_types[i]);
		}
		aggregates = AggregateObject::CreateAggregateObjects(aggr_bindings);
		for (idx_t i = 0; i < aggregates.size(); i++) {
			payload_width += aggregates[i].payload_size;
		}
		empty_payload_data = unique_ptr<data_t[]>(new data_t[payload_width]);
		// initialize the aggregates to the NULL value
		auto pointer = empty_payload_data.get();
		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto &aggr = aggregates[i];
			aggr.function.initialize(pointer);
			pointer += aggr.payload_size;
		}
		tuple_size = sizeof(hash_t) + group_width + payload_width;
	}

	//! Expression executor for the GROUP BY chunk
	ExpressionExecutor group_executor;
	//! Expression state for the payload
	ExpressionExecutor payload_executor;
	//! Materialized GROUP BY expression
	DataChunk group_chunk;
	//! The payload chunk
	DataChunk payload_chunk;




	vector<AggregateObject> aggregates;

	idx_t group_width;
	idx_t payload_width;
	unique_ptr<data_t[]> empty_payload_data;

	//! The size of the initial flag for each cell
	static constexpr int FLAG_SIZE = sizeof(uint8_t);
	//! Flag indicating a cell is empty
	static constexpr int EMPTY_CELL = 0x00;
	//! Flag indicating a cell is full
	static constexpr int FULL_CELL = 0xFF;

	idx_t capacity = 2048;
    idx_t entries = 0;

	struct table1_entry_t {
		uint8_t flag;
		data_ptr_t entry_ptr;  // TODO maybe replace this with buffer mgr id + offset
	};

	uint64_t bitmask = capacity - 1;

	unique_ptr<BufferHandle> table1;

	idx_t tuple_size;

	// TODO this is not buffer managed!!
    StringHeap string_heap;
    data_ptr_t endptr;
    data_ptr_t data;
    data_ptr_t data_t2;
    idx_t pos_t2;

    template <class T>
    static void templated_scatter(VectorData &gdata, Vector &addresses, const SelectionVector &sel, idx_t count,
                                  idx_t type_size) {
        auto data = (T *)gdata.data;
        auto pointers = FlatVector::GetData<uintptr_t>(addresses);
        if (gdata.nullmask->any()) {
            for (idx_t i = 0; i < count; i++) {
                auto pointer_idx = sel.get_index(i);
                auto group_idx = gdata.sel->get_index(pointer_idx);
                auto ptr = (T *)pointers[pointer_idx];

                if ((*gdata.nullmask)[group_idx]) {
                    *ptr = NullValue<T>();
                } else {
                    *ptr = data[group_idx];
                }
                pointers[pointer_idx] += type_size;
            }
        } else {
            for (idx_t i = 0; i < count; i++) {
                auto pointer_idx = sel.get_index(i);
                auto group_idx = gdata.sel->get_index(pointer_idx);
                auto ptr = (T *)pointers[pointer_idx];

                *ptr = data[group_idx];
                pointers[pointer_idx] += type_size;
            }
        }
    }

    void ScatterGroups(DataChunk &groups, unique_ptr<VectorData[]> &group_data, Vector &addresses,
                                            const SelectionVector &sel, idx_t count) {
        for (idx_t grp_idx = 0; grp_idx < groups.column_count(); grp_idx++) {
            auto &data = groups.data[grp_idx];
            auto &gdata = group_data[grp_idx];

            auto type_size = GetTypeIdSize(data.type);

            switch (data.type) {
            case TypeId::BOOL:
            case TypeId::INT8:
                templated_scatter<int8_t>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::INT16:
                templated_scatter<int16_t>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::INT32:
                templated_scatter<int32_t>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::INT64:
                templated_scatter<int64_t>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::FLOAT:
                templated_scatter<float>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::DOUBLE:
                templated_scatter<double>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::INTERVAL:
                templated_scatter<interval_t>(gdata, addresses, sel, count, type_size);
                break;
            case TypeId::VARCHAR: {
                auto data = (string_t *)gdata.data;
                auto pointers = FlatVector::GetData<uintptr_t>(addresses);

                for (idx_t i = 0; i < count; i++) {
                    auto pointer_idx = sel.get_index(i);
                    auto group_idx = gdata.sel->get_index(pointer_idx);
                    auto ptr = (string_t *)pointers[pointer_idx];

                    if ((*gdata.nullmask)[group_idx]) {
                        *ptr = NullValue<string_t>();
                    } else if (data[group_idx].IsInlined()) {
                        *ptr = data[group_idx];
                    } else {
                        *ptr = string_heap.AddString(data[group_idx]);
                    }
                    pointers[pointer_idx] += type_size;
                }
                break;
            }
            default:
                throw Exception("Unsupported type for group vector");
            }
        }
    }



    idx_t FindOrCreateGroups(DataChunk &groups, Vector &addresses, SelectionVector &new_groups) {
        // resize at 50% capacity, also need to fit the entire vector
        if (entries > capacity / 2 || capacity - entries <= STANDARD_VECTOR_SIZE) {
            //Resize(capacity * 2);
            assert(0);
		}

        // we need to be able to fit at least one vector of data
        assert(capacity - entries > STANDARD_VECTOR_SIZE);
        assert(addresses.type == TypeId::POINTER);

        // hash the groups to get the addresses
        Vector hashes(TypeId::HASH);
		auto hashes_ptr = FlatVector::GetData<hash_t>(hashes);
        groups.Hash(hashes);

        // now compute the entry in the table based on the hash using a modulo
        // multiply the position by the tuple size and add the base address
        UnaryExecutor::Execute<hash_t, data_ptr_t>(hashes, addresses, groups.size(), [&](hash_t element) {
          assert((element & bitmask) == (element % capacity));
          return table1->node->buffer + ((element & bitmask) * sizeof(table1_entry_t));
        });

        addresses.Normalify(groups.size());
        auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);

        data_ptr_t group_pointers[STANDARD_VECTOR_SIZE];
        Vector group_pointers_vector(TypeId::POINTER, (data_ptr_t)group_pointers);

        // set up the selection vectors
        SelectionVector v1(STANDARD_VECTOR_SIZE);
        SelectionVector v2(STANDARD_VECTOR_SIZE);
        SelectionVector empty_vector(STANDARD_VECTOR_SIZE);

        // we start out with all entries [0, 1, 2, ..., groups.size()]
        const SelectionVector *sel_vector = &FlatVector::IncrementalSelectionVector;
        SelectionVector *next_vector = &v1;
        SelectionVector *no_match_vector = &v2;
        idx_t remaining_entries = groups.size();

        // orrify all the groups
        auto group_data = unique_ptr<VectorData[]>(new VectorData[groups.column_count()]);
        for (idx_t grp_idx = 0; grp_idx < groups.column_count(); grp_idx++) {
            groups.data[grp_idx].Orrify(groups.size(), group_data[grp_idx]);
        }

        idx_t new_group_count = 0;
        while (remaining_entries > 0) {
            idx_t entry_count = 0;
            idx_t empty_count = 0;

            // first figure out for each remaining whether or not it belongs to a full or empty group
            for (idx_t i = 0; i < remaining_entries; i++) {
                idx_t index = sel_vector->get_index(i);
                auto entry = data_pointers[index];
                if (*entry == EMPTY_CELL) {
                    // cell is empty; mark the cell as filled
                    *entry = FULL_CELL;
                    empty_vector.set_index(empty_count++, index);
                    new_groups.set_index(new_group_count++, index);
                    auto t1_ptr = (table1_entry_t*) entry;
					t1_ptr->entry_ptr = data_t2;
					data_t2 += tuple_size;
					*((hash_t*) t1_ptr->entry_ptr) = hashes_ptr[index];
                    // initialize the payload info for the column
                    memcpy(t1_ptr->entry_ptr + sizeof(hash_t) + group_width, empty_payload_data.get(), payload_width);
                } else {
                    // cell is occupied: add to check list
                    next_vector->set_index(entry_count++, index);
                }
                group_pointers[index] = ((table1_entry_t*) entry)->entry_ptr + sizeof(hash_t);
                data_pointers[index] = entry + sizeof(table1_entry_t);
            }
            group_pointers_vector.Print(groups.size());

            if (empty_count > 0) {
                // for each of the locations that are empty, serialize the group columns to the locations
                ScatterGroups(groups, group_data, group_pointers_vector, empty_vector, empty_count);
                entries += empty_count;
            }
            // now we have only the tuples remaining that might match to an existing group
            // start performing comparisons with each of the groups
            idx_t no_match_count = CompareGroups(groups, group_data, group_pointers_vector, *next_vector, entry_count, *no_match_vector);

            // each of the entries that do not match we move them to the next entry in the HT
            for (idx_t i = 0; i < no_match_count; i++) {
                idx_t index = no_match_vector->get_index(i);
                data_pointers[index] += payload_width;
                assert(((uint64_t)(data_pointers[index] - data)) % tuple_size == 0);
                if (data_pointers[index] >= endptr) {
                    data_pointers[index] = data;
                }
            }
            sel_vector = no_match_vector;
            std::swap(next_vector, no_match_vector);
            remaining_entries = no_match_count;
        }
        return new_group_count;
    }


    template <class T>
    static void templated_compare_groups(VectorData &gdata, Vector &addresses, SelectionVector &sel, idx_t &count,
                                         idx_t type_size, SelectionVector &no_match, idx_t &no_match_count) {
        auto data = (T *)gdata.data;
        auto pointers = FlatVector::GetData<uintptr_t>(addresses);
        idx_t match_count = 0;
        if (gdata.nullmask->any()) {
            for (idx_t i = 0; i < count; i++) {
                auto idx = sel.get_index(i);
                auto group_idx = gdata.sel->get_index(idx);
                auto value = (T *)pointers[idx];

                if ((*gdata.nullmask)[group_idx]) {
                    if (IsNullValue<T>(*value)) {
                        // match: move to next value to compare
                        sel.set_index(match_count++, idx);
                        pointers[idx] += type_size;
                    } else {
                        no_match.set_index(no_match_count++, idx);
                    }
                } else {
                    if (Equals::Operation<T>(data[group_idx], *value)) {
                        sel.set_index(match_count++, idx);
                        pointers[idx] += type_size;
                    } else {
                        no_match.set_index(no_match_count++, idx);
                    }
                }
            }
        } else {
            for (idx_t i = 0; i < count; i++) {
                auto idx = sel.get_index(i);
                auto group_idx = gdata.sel->get_index(idx);
                auto value = (T *)pointers[idx];

                if (Equals::Operation<T>(data[group_idx], *value)) {
                    sel.set_index(match_count++, idx);
                    pointers[idx] += type_size;
                } else {
                    no_match.set_index(no_match_count++, idx);
                }
            }
        }
        count = match_count;
    }

    static idx_t CompareGroups(DataChunk &groups, unique_ptr<VectorData[]> &group_data, Vector &addresses,
                               SelectionVector &sel, idx_t count, SelectionVector &no_match) {
        idx_t no_match_count = 0;
        for (idx_t group_idx = 0; group_idx < groups.column_count(); group_idx++) {
            auto &data = groups.data[group_idx];
            auto &gdata = group_data[group_idx];
            auto type_size = GetTypeIdSize(data.type);
            switch (data.type) {
            case TypeId::BOOL:
            case TypeId::INT8:
                templated_compare_groups<int8_t>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::INT16:
                templated_compare_groups<int16_t>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::INT32:
                templated_compare_groups<int32_t>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::INT64:
                templated_compare_groups<int64_t>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::FLOAT:
                templated_compare_groups<float>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::DOUBLE:
                templated_compare_groups<double>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::INTERVAL:
                templated_compare_groups<interval_t>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            case TypeId::VARCHAR:
                templated_compare_groups<string_t>(gdata, addresses, sel, count, type_size, no_match, no_match_count);
                break;
            default:
                throw Exception("Unsupported type for group vector");
            }
        }
        return no_match_count;
    }

	void AddChunk(BufferManager &bm) {
		// pointers:
		// [FLAG][BLOCK][OFFSET]

		Vector addresses(TypeId::POINTER);
		SelectionVector new_groups(STANDARD_VECTOR_SIZE);

		// TODO: needs to be in init
		table1 = bm.Allocate(std::max((idx_t)Storage::BLOCK_ALLOC_SIZE, capacity * (sizeof(table1_entry_t))));

		// zero memory!
		memset(table1->node->buffer, 0, table1->node->size);
        data = table1->node->buffer;
        endptr = data + capacity * sizeof(table1_entry_t);

        auto table2 = bm.Allocate(Storage::BLOCK_ALLOC_SIZE);
        data_t2 = table2->node->buffer;

		// figure out groups

        FindOrCreateGroups(group_chunk,addresses, new_groups );

        idx_t payload_idx = 0;

        for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
            assert(payload_chunk.column_count() > payload_idx);

            // for any entries for which a group was found, update the aggregate
            auto &aggr = aggregates[aggr_idx];
            auto input_count = max((idx_t)1, (idx_t)aggr.child_count);

			aggr.function.update(&payload_chunk.data[payload_idx], input_count, addresses, payload_chunk.size());


            // move to the next aggregate
            payload_idx += input_count;
            VectorOperations::AddInPlace(addresses, aggr.payload_size, payload_chunk.size());
        }

		// payload
		// [HASH][GROUPS][PAYLOAD]
		// [HASH] is the hash of the groups
		// [GROUPS] is the groups
		// [PAYLOAD] is the payload (i.e. the aggregate states)
	}
};

unique_ptr<GlobalOperatorState> PhysicalHashAggregate::GetGlobalState(ClientContext &context) {
	return make_unique<HashAggregateGlobalState>(group_types, payload_types, bindings);
}

unique_ptr<LocalSinkState> PhysicalHashAggregate::GetLocalSinkState(ExecutionContext &context) {
	return make_unique<HashAggregateLocalState>(groups, bindings, group_types, payload_types);
}

void PhysicalHashAggregate::Sink(ExecutionContext &context, GlobalOperatorState &state, LocalSinkState &lstate,
                                 DataChunk &input) {
	auto &gstate = (HashAggregateGlobalState &)state;
	auto &sink = (HashAggregateLocalState &)lstate;

	DataChunk &group_chunk = sink.group_chunk;
	DataChunk &payload_chunk = sink.payload_chunk;
	sink.group_executor.Execute(input, group_chunk);
	sink.payload_executor.SetChunk(input);

	payload_chunk.Reset();
	idx_t payload_idx = 0, payload_expr_idx = 0;
	payload_chunk.SetCardinality(group_chunk);
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = (BoundAggregateExpression &)*aggregates[i];
		if (aggr.children.size()) {
			for (idx_t j = 0; j < aggr.children.size(); ++j) {
				sink.payload_executor.ExecuteExpression(payload_expr_idx, payload_chunk.data[payload_idx]);
				payload_idx++;
				payload_expr_idx++;
			}
		} else {
			payload_idx++;
		}
	}

	group_chunk.Verify();
	payload_chunk.Verify();
	assert(payload_chunk.column_count() == 0 || group_chunk.size() == payload_chunk.size());

	group_chunk.Print();
	payload_chunk.Print();

	sink.AddChunk(*context.client.db.storage->buffer_manager);

	// TODO is this slow? should probably be in local state and or-ed on combine!
	gstate.is_empty = false;
}

//===--------------------------------------------------------------------===//
// GetChunkInternal
//===--------------------------------------------------------------------===//
class PhysicalHashAggregateState : public PhysicalOperatorState {
public:
	PhysicalHashAggregateState(vector<TypeId> &group_types, vector<TypeId> &aggregate_types, PhysicalOperator *child)
	    : PhysicalOperatorState(child), ht_scan_position(0) {
		group_chunk.Initialize(group_types);
		if (aggregate_types.size() > 0) {
			aggregate_chunk.Initialize(aggregate_types);
		}
	}

	//! Materialized GROUP BY expression
	DataChunk group_chunk;
	//! Materialized aggregates
	DataChunk aggregate_chunk;
	//! The current position to scan the HT for output tuples
	idx_t ht_scan_position;
};

void PhysicalHashAggregate::GetChunkInternal(ExecutionContext &context, DataChunk &chunk,
                                             PhysicalOperatorState *state_) {
	//	auto &gstate = (HashAggregateGlobalState &)*sink_state;
	auto &state = (PhysicalHashAggregateState &)*state_;
	//
	//	state.group_chunk.Reset();
	//	state.aggregate_chunk.Reset();
	//	idx_t elements_found = gstate.ht->Scan(state.ht_scan_position, state.group_chunk, state.aggregate_chunk);
	//
	//	// special case hack to sort out aggregating from empty intermediates
	//	// for aggregations without groups
	//	if (elements_found == 0 && gstate.is_empty && is_implicit_aggr) {
	//		assert(chunk.column_count() == aggregates.size());
	//		// for each column in the aggregates, set to initial state
	//		chunk.SetCardinality(1);
	//		for (idx_t i = 0; i < chunk.column_count(); i++) {
	//			assert(aggregates[i]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
	//			auto &aggr = (BoundAggregateExpression &)*aggregates[i];
	//			auto aggr_state = unique_ptr<data_t[]>(new data_t[aggr.function.state_size()]);
	//			aggr.function.initialize(aggr_state.get());
	//
	//			Vector state_vector(Value::POINTER((uintptr_t)aggr_state.get()));
	//			aggr.function.finalize(state_vector, chunk.data[i], 1);
	//		}
	//		state.finished = true;
	//		return;
	//	}
	//	if (elements_found == 0 && !state.finished) {
	//		state.finished = true;
	//		return;
	//	}
	//	// compute the final projection list
	//	idx_t chunk_index = 0;
	//	chunk.SetCardinality(elements_found);
	//	if (state.group_chunk.column_count() + state.aggregate_chunk.column_count() == chunk.column_count()) {
	//		for (idx_t col_idx = 0; col_idx < state.group_chunk.column_count(); col_idx++) {
	//			chunk.data[chunk_index++].Reference(state.group_chunk.data[col_idx]);
	//		}
	//	} else {
	//		assert(state.aggregate_chunk.column_count() == chunk.column_count());
	//	}
	//
	//	for (idx_t col_idx = 0; col_idx < state.aggregate_chunk.column_count(); col_idx++) {
	//		chunk.data[chunk_index++].Reference(state.aggregate_chunk.data[col_idx]);
	//	}
	state.finished = true;
	return;
}

unique_ptr<PhysicalOperatorState> PhysicalHashAggregate::GetOperatorState() {
	return make_unique<PhysicalHashAggregateState>(group_types, aggregate_types,
	                                               children.size() == 0 ? nullptr : children[0].get());
}
