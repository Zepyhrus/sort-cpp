# CMakeLists from src directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

file(GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

# using opencv
find_package(OpenCV REQUIRED)
message(STATUS "OPENCV libs: " ${OpenCV_LIBS})

# using gflag
find_package(gflags REQUIRED)

add_executable(SORT main.cpp Hungarian.cpp KalmanTracker.cpp)


target_link_libraries(SORT ${OpenCV_LIBS})
target_link_libraries(SORT stdc++fs)
target_link_libraries(SORT gflags)