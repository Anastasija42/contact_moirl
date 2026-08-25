cmake_minimum_required(VERSION 3.10)
project(friction_lib)

set(CMAKE_CXX_STANDARD 14)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(crocoddyl REQUIRED)
find_package(pinocchio REQUIRED)
find_package(eigenpy REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

add_library(friction_lib SHARED friction_model.cpp) 

# --- CRITICAL FIX: Pass compile definitions from Pinocchio ---
target_compile_definitions(friction_lib PRIVATE ${pinocchio_DEFINITIONS})
# -------------------------------------------------------------

target_include_directories(friction_lib PRIVATE
    ${crocoddyl_INCLUDE_DIRS}
    ${pinocchio_INCLUDE_DIRS}
    ${eigenpy_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(friction_lib PRIVATE
    crocoddyl::crocoddyl
    pinocchio::pinocchio
    eigenpy::eigenpy
    ${Boost_PYTHON_LIBRARIES}
)

set_target_properties(friction_lib PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(friction_lib PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(friction_lib PROPERTIES SUFFIX ".so")
endif()