# CMake generated Testfile for 
# Source directory: /gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Utilities/cmcurl
# Build directory: /gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[curl]=] "curltest" "http://open.cdash.org/user.php")
set_tests_properties([=[curl]=] PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Utilities/cmcurl/CMakeLists.txt;1580;add_test;/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Utilities/cmcurl/CMakeLists.txt;0;")
subdirs("lib")
