# create the binary directory
make_directory("/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/CMakeBuildCOnly")

# remove the CMakeCache.txt file from the source dir
# if there is one, so that in-source cmake tests
# still pass
message("Remove: /gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/COnly/CMakeCache.txt")
file(REMOVE "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/COnly/CMakeCache.txt")

# run cmake in the binary directory
message("running: ${CMAKE_COMMAND}")
execute_process(COMMAND "${CMAKE_COMMAND}"
  "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/COnly"
  "-GUnix Makefiles"
  -A ""
  -T ""
  WORKING_DIRECTORY "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/CMakeBuildCOnly"
  RESULT_VARIABLE RESULT)
if(RESULT)
  message(FATAL_ERROR "Error running cmake command")
endif()

# Now use the --build option to build the project
message("running: ${CMAKE_COMMAND} --build")
execute_process(COMMAND "${CMAKE_COMMAND}"
  --build "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/CMakeBuildCOnly" --config Debug
  RESULT_VARIABLE RESULT)
if(RESULT)
  message(FATAL_ERROR "Error running cmake --build")
endif()

# run the executable out of the Debug directory if using a
# multi-config generator
set(_isMultiConfig 0)
if(_isMultiConfig)
  set(RUN_TEST "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/CMakeBuildCOnly/Debug/COnly")
else()
  set(RUN_TEST "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/CMakeBuildCOnly/COnly")
endif()
# run the test results
message("running [${RUN_TEST}]")
execute_process(COMMAND "${RUN_TEST}" RESULT_VARIABLE RESULT)
if(RESULT)
  message(FATAL_ERROR "Error running test COnly")
endif()

# build it again with clean and only COnly target
execute_process(COMMAND "${CMAKE_COMMAND}"
  --build "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/CMakeBuildCOnly" --config Debug
  --clean-first --target COnly
  RESULT_VARIABLE RESULT)
if(RESULT)
  message(FATAL_ERROR "Error running cmake --build")
endif()

# run it again after clean
execute_process(COMMAND "${RUN_TEST}" RESULT_VARIABLE RESULT)
if(RESULT)
  message(FATAL_ERROR "Error running test COnly after clean ")
endif()
