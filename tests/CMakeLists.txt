# tests/CMakeLists.txt — 单测框架接入 ctest
# 断言宏在 tests/test_util.h (header-only, 无第三方依赖)

function(gsv_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE GSV_WEIGHTS_DIR="${GSV_WEIGHTS_DIR}"
                                             GSV_KERN_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kern_golden")
  target_link_libraries(${name} PRIVATE gsv_rt gsv_kern)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

gsv_add_test(test_gsv_header)
gsv_add_test(test_gsv_loader)
gsv_add_test(test_kern)
gsv_add_test(test_accel)

# 层验证 harness(由 tools/layer0_check.py 驱动, 不注册 ctest)
add_executable(layer0_check layer0_check.cpp)
target_link_libraries(layer0_check PRIVATE gsv_rt gsv_kern)
