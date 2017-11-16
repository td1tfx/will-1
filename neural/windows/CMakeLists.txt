project(will-windows)

aux_source_directory(./ SRC_LIST)

add_executable(will-windows ${SRC_LIST})
target_link_libraries(will-windows will)
