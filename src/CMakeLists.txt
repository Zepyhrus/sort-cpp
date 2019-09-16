# CMakeLists from src directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

file(GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

find_package(OpenCV REQUIRED)
message(STATUS "OPENCV libs: " ${OpenCV_LIBS})

add_executable(SORT ${srcs})


target_link_libraries(SORT ${OpenCV_LIBS})
target_link_libraries(SORT stdc++fs)