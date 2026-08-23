cmake_minimum_required(VERSION 3.25)
project(gsv-native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT APPLE)
  message(FATAL_ERROR "gsv-native 仅面向 Apple Silicon (纯 CPU + Accelerate)")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
  message(WARNING "非 Apple clang 未验证; 继续, 但数值/性能基线以 AppleClang 为准")
endif()

# ---- 编译选项 ----
# 全配置: M4 目标特性 (FEAT_FP16/FMLAL 等); 零警告纪律
add_compile_options(-mcpu=apple-m4 -Wall -Wextra -Wpedantic -Wshadow -Werror)
# Release: 任务卡指定 -O3 -fvectorize (显式写出, 不依赖 CMake 默认串)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fvectorize")
# Debug 变体保留: 无优化 + 全量调试信息
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -fvectorize")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- 运行时库 ----
add_library(gsv_rt STATIC
  src/runtime/gsv_header.cpp
  src/runtime/gsv_loader.cpp
  src/runtime/mini_json.cpp
  src/runtime/threadpool.cpp
)
add_library(gsv_kern STATIC
  src/kern/kern.cpp
  src/kern/gemv.cpp
  src/kern/accel.cpp
)
target_link_libraries(gsv_kern PUBLIC "-framework Accelerate")

# ---- AR/T2S 引擎 (task/B12) ----
add_library(gsv_ar STATIC
  src/ar/t2s_engine.cpp
)
target_link_libraries(gsv_ar PUBLIC gsv_rt gsv_kern)
# macOS 13.3+ 旧 cblas 接口标记弃用; 启用新头(不带 ILP64, 保持 32 位整型约定)
target_compile_definitions(gsv_kern PUBLIC ACCELERATE_NEW_LAPACK)

# ---- 编码器 (task/C1: HuBERT + SV + fbank + wav + ref cache) ----
add_library(gsv_encoder STATIC
  src/runtime/wav.cpp
  src/encoder/fbank.cpp
  src/encoder/hubert.cpp
  src/encoder/sv.cpp
  src/encoder/ref_cache.cpp
)
target_link_libraries(gsv_encoder PUBLIC gsv_rt gsv_kern)

# ---- SoVITS (B34: header-only 引擎 + 类型/工具) ----
add_library(gsv_sovits INTERFACE)
target_include_directories(gsv_sovits INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- 文本前端 (B9/M3: jieba+g2p+归一化) ----
add_library(gsv_textfront STATIC
  src/textfront/jieba.cpp
  src/textfront/pinyin.cpp
  src/textfront/tone_sandhi.cpp
  src/textfront/zh_norm.cpp
  src/textfront/zh_rules.cpp
  src/textfront/zh_verbalize.cpp
  src/textfront/chinese_g2p.cpp
  src/textfront/textfront.cpp
)

# ---- CLI 工具 ----
add_executable(gsv_tool src/runtime/gsv_tool.cpp)
target_link_libraries(gsv_tool PRIVATE gsv_rt)

# ---- 全链路 CLI (task/C2) ----
add_executable(gsv_native
  src/main.cpp
  src/runtime/pipeline.cpp
  src/runtime/pipeline_condition.cpp
)
target_link_libraries(gsv_native PRIVATE gsv_rt gsv_kern gsv_ar gsv_encoder
gsv_textfront gsv_sovits)

# ---- 测试 ----
enable_testing()
set(GSV_WEIGHTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/weights" CACHE PATH "weights 目录")
add_subdirectory(tests)
