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

# E8: BERT AMX 路径一致性(无权重/无AMX时 SKIP; 有权重时跑数值对照)
gsv_add_test(test_bert_amx)

# 层验证 harness(由 tools/layer0_check.py 驱动, 不注册 ctest)
add_executable(layer0_check layer0_check.cpp)
target_link_libraries(layer0_check PRIVATE gsv_rt gsv_kern)

# ---- task/B12: AR 引擎单测与 pairs 验收 harness(本地验证用) ----
add_executable(test_ar_unit test_ar_unit.cpp)
target_include_directories(test_ar_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_ar_unit PRIVATE
  GSV_WEIGHTS_DIR="${GSV_WEIGHTS_DIR}"
  GSV_KERN_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kern_golden"
  GSV_AR_RAW_DIR="${CMAKE_CURRENT_SOURCE_DIR}/local/golden_raw")
target_link_libraries(test_ar_unit PRIVATE gsv_ar)
add_test(NAME test_ar_unit COMMAND test_ar_unit)

add_executable(ar_pair_run ar_pair_run.cpp)
target_link_libraries(ar_pair_run PRIVATE gsv_ar)

# ---- task/C1: 编码器单测与 fixtures 验收 harness(本地验证用) ----
add_executable(test_c1_unit test_c1_unit.cpp)
target_compile_definitions(test_c1_unit PRIVATE
  GSV_WEIGHTS_DIR="${GSV_WEIGHTS_DIR}"
  C1_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/local/c1_fixtures")
target_link_libraries(test_c1_unit PRIVATE gsv_encoder)
add_test(NAME test_c1_unit COMMAND test_c1_unit)

add_executable(c1_run c1_run.cpp)
target_compile_definitions(c1_run PRIVATE
  GSV_WEIGHTS_DIR="${GSV_WEIGHTS_DIR}"
  C1_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/local/c1_fixtures")
target_link_libraries(c1_run PRIVATE gsv_encoder)

# ---- task/E2-ENC: fp16 化验收 harness(本地跨分支对照用, 不注册 ctest) ----
add_executable(e2_enc_dump local/e2_enc_dump.cpp)
target_link_libraries(e2_enc_dump PRIVATE gsv_rt gsv_kern gsv_encoder)

add_executable(e2_hub_layers local/e2_hub_layers.cpp)
target_link_libraries(e2_hub_layers PRIVATE gsv_rt gsv_kern gsv_encoder)

add_executable(e2_sv_layers local/e2_sv_layers.cpp)
target_link_libraries(e2_sv_layers PRIVATE gsv_rt gsv_kern gsv_encoder)

# c2_pairs_run(全链路 phones/tokens 与 pairs golden 对照; 需 data/weights)
add_executable(c2_pairs_run local/c2_pairs_run.cpp)
target_sources(c2_pairs_run PRIVATE ${CMAKE_SOURCE_DIR}/src/runtime/pipeline.cpp
  ${CMAKE_SOURCE_DIR}/src/runtime/pipeline_condition.cpp)
target_link_libraries(c2_pairs_run PRIVATE gsv_rt gsv_kern gsv_ar gsv_encoder
gsv_textfront gsv_sovits)

add_executable(e2_g2pw_all local/e2_g2pw_all.cpp)
target_link_libraries(e2_g2pw_all PRIVATE gsv_rt gsv_kern gsv_textfront)

add_executable(e2_bench local/e2_bench.cpp)
target_link_libraries(e2_bench PRIVATE gsv_rt gsv_kern gsv_encoder)
