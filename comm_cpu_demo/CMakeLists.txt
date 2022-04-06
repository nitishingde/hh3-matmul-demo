add_executable(comm_cpu_demo
		comm_cpu_demo.cc

		data/data_type.h
		data/matrix_block_data.h
		data/matrix_data.h

		state/input_block_state.h
		state/output_state.h
		state/partial_computation_state.h
		state/partial_computation_state_manager.h

		task/addition_task.h
		task/product_task.h
		task/matrix_row_traversal_task.h
		task/matrix_column_traversal_task.h

		${tclap_headers}
		)

target_link_libraries(comm_cpu_demo ${OpenBLAS_LIBRARIES})
target_include_directories(comm_cpu_demo PUBLIC ${OpenBLAS_INCLUDE_DIRS})
target_compile_definitions(comm_cpu_demo PUBLIC -DUSE_OPENBLAS)

target_link_libraries(comm_cpu_demo ${Hedgehog_LIBRARIES})
target_include_directories(comm_cpu_demo PUBLIC ${Hedgehog_INCLUDE_DIRS})