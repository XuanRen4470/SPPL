# CMake generated Testfile for 
# Source directory: /gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4
# Build directory: /gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/Tests/EnforceConfig.cmake")
add_test([=[SystemInformationNew]=] "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/bin/cmake" "--system-information" "-G" "Unix Makefiles")
set_tests_properties([=[SystemInformationNew]=] PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/CMakeLists.txt;519;add_test;/gpfs/users/a1796450/ACL_2024/Minimum_Change/downward/cmake-3.27.4/CMakeLists.txt;0;")
subdirs("Source/kwsys")
subdirs("Utilities/std")
subdirs("Utilities/KWIML")
subdirs("Utilities/cmlibrhash")
subdirs("Utilities/cmzlib")
subdirs("Utilities/cmcurl")
subdirs("Utilities/cmnghttp2")
subdirs("Utilities/cmexpat")
subdirs("Utilities/cmbzip2")
subdirs("Utilities/cmzstd")
subdirs("Utilities/cmliblzma")
subdirs("Utilities/cmlibarchive")
subdirs("Utilities/cmjsoncpp")
subdirs("Utilities/cmlibuv")
subdirs("Utilities/cmcppdap")
subdirs("Source")
subdirs("Utilities")
subdirs("Tests")
subdirs("Auxiliary")
