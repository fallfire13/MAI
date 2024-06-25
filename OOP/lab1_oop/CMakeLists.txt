project(lab1)

add_executable(lab1
  lab1.cpp
  complex.cpp)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Wextra")