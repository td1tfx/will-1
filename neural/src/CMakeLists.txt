project(libwill)

aux_source_directory(./ SRC_LIST)
aux_source_directory(./others OTHER_SRC_LIST)
list(APPEND SRC_LIST ${OTHER_SRC_LIST})

set(LINK_LIBRARIES will-cuda cudnn cudart curand cublas openblas)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
  list(APPEND LINK_LIBRARIES nvidia-ml)
endif()

add_library(will SHARED ${SRC_LIST})
target_link_libraries(will ${LINK_LIBRARIES})
