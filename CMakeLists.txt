cmake_minimum_required(VERSION 3.16)
project(ipos3_opt LANGUAGES CXX)

# Ceres 2.2 需要 C++17；本工程只依赖 Eigen + Ceres
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)   # MinGW 下需要 gnu++17（严格 ANSI 会缺 j0/j1/jn）

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# 依赖
#   Ubuntu:  sudo apt install libeigen3-dev libceres-dev     (Ceres 需 >= 2.1)
#   自编译 Ceres 时若用 MINIGLOG，需要额外把 <ceres>/internal/ceres/miniglog
#   加进包含路径（见下面注释掉的那一行）。
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# ---------------------------------------------------------------------------
# 算法库
# ---------------------------------------------------------------------------
add_library(ipos3_opt_lib STATIC src/alg_lib.cpp)
target_include_directories(ipos3_opt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_include_directories(ipos3_opt_lib PUBLIC <CERES_SOURCE>/internal/ceres/miniglog)
target_link_libraries(ipos3_opt_lib PUBLIC Eigen3::Eigen)

if(TARGET Ceres::ceres)
    target_link_libraries(ipos3_opt_lib PUBLIC Ceres::ceres)
else()
    target_include_directories(ipos3_opt_lib PUBLIC ${CERES_INCLUDE_DIRS})
    target_link_libraries(ipos3_opt_lib PUBLIC ${CERES_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# 可执行程序
# ---------------------------------------------------------------------------
add_executable(ipos3_opt src/main.cpp)
target_link_libraries(ipos3_opt PRIVATE ipos3_opt_lib)


add_executable(test_math test/test_math.cpp)
target_link_libraries(test_math PRIVATE ipos3_opt_lib)

add_executable(test_pre test/test_pre.cpp)
target_link_libraries(test_pre PRIVATE ipos3_opt_lib)

add_executable(test_ma test/test_ma.cpp)
target_link_libraries(test_ma PRIVATE ipos3_opt_lib)

add_executable(test_marg test/test_marg.cpp)
target_link_libraries(test_marg PRIVATE ipos3_opt_lib)

#add_executable(test_flow test/test_flow.cpp)
#target_link_libraries(test_flow PRIVATE ipos3_opt_lib)