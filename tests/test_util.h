// test_util.h — header-only 单测断言宏与极简 runner（无第三方依赖）
//
// 用法:
//   GSV_TEST(arith) { CHECK(1+1==2); CHECK_NEAR(0.1+0.2, 0.3, 1e-9); }
//   GSV_TEST_MAIN()
#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gsvtest {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

inline int& fail_count() {
  static int n = 0;
  return n;
}

// 每个用例独立计数, 供 runner 打 PASS/FAIL
inline int& current_case_failures() {
  static int n = 0;
  return n;
}

template <typename T>
void check_impl(bool ok, const T& expr_text, const char* file, int line) {
  if (!ok) {
    ++fail_count();
    ++current_case_failures();
    std::printf("    CHECK 失败: %s @ %s:%d\n", expr_text, file, line);
  }
}

inline void check_near_impl(double a, double b, double tol, const char* ea, const char* eb,
                            const char* etol, const char* file, int line) {
  if (!(std::fabs(a - b) <= tol)) {
    ++fail_count();
    ++current_case_failures();
    std::printf("    CHECK_NEAR 失败: |%s(%g) - %s(%g)| > %s(%g) @ %s:%d\n", ea, a, eb, b, etol,
                tol, file, line);
  }
}

inline int run_all() {
  for (const auto& tc : registry()) {
    const int before = fail_count();
    current_case_failures() = 0;
    tc.fn();
    std::printf("[%s] %s\n", fail_count() == before ? "PASS" : "FAIL", tc.name);
  }
  const int total_fail = fail_count();
  std::printf("== %zu 个用例, %d 失败 ==\n", registry().size(), total_fail);
  return total_fail == 0 ? 0 : 1;
}

}  // namespace gsvtest

#define GSV_TEST(name)                                                        \
  static void gsv_test_fn_##name();                                           \
  static const bool gsv_test_reg_##name = [] {                                \
    ::gsvtest::registry().push_back({#name, &gsv_test_fn_##name});            \
    return true;                                                              \
  }();                                                                        \
  static void gsv_test_fn_##name()

#define GSV_TEST_MAIN()                              \
  int main() {                                       \
    return ::gsvtest::run_all();                     \
  }

#define CHECK(cond) ::gsvtest::check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_MSG(cond, msg)                                     \
  do {                                                           \
    if (!static_cast<bool>(cond)) std::printf("    note: %s\n", msg); \
    ::gsvtest::check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__); \
  } while (0)
#define CHECK_EQ(a, b) ::gsvtest::check_impl((a) == (b), #a " == " #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
  ::gsvtest::check_near_impl(a, b, tol, #a, #b, #tol, __FILE__, __LINE__)
