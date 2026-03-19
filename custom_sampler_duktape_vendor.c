#include "vendor/duktape/duktape.h"

#ifndef DUK_USE_DATE_GET_NOW
#if defined(DUK_USE_DATE_NOW_GETTIMEOFDAY)
#define DUK_USE_DATE_GET_NOW(ctx)            duk_bi_date_get_now_gettimeofday()
#elif defined(DUK_USE_DATE_NOW_TIME)
#define DUK_USE_DATE_GET_NOW(ctx)            duk_bi_date_get_now_time()
#elif defined(DUK_USE_DATE_NOW_WINDOWS)
#define DUK_USE_DATE_GET_NOW(ctx)            duk_bi_date_get_now_windows()
#elif defined(DUK_USE_DATE_NOW_WINDOWS_SUBMS)
#define DUK_USE_DATE_GET_NOW(ctx)            duk_bi_date_get_now_windows_subms()
#endif
#endif

#ifndef DUK_USE_DATE_GET_LOCAL_TZOFFSET
#if defined(DUK_USE_DATE_TZO_GMTIME_R)
#define DUK_USE_DATE_GET_LOCAL_TZOFFSET(d)   duk_bi_date_get_local_tzoffset_gmtime((d))
#elif defined(DUK_USE_DATE_TZO_WINDOWS)
#define DUK_USE_DATE_GET_LOCAL_TZOFFSET(d)   duk_bi_date_get_local_tzoffset_windows((d))
#elif defined(DUK_USE_DATE_TZO_WINDOWS_NO_DST)
#define DUK_USE_DATE_GET_LOCAL_TZOFFSET(d)   duk_bi_date_get_local_tzoffset_windows_no_dst((d))
#endif
#endif

#undef DUK_USE_INTERRUPT_COUNTER
#define DUK_USE_INTERRUPT_COUNTER

#undef DUK_USE_EXEC_TIMEOUT_CHECK
#define DUK_USE_EXEC_TIMEOUT_CHECK(udata) kcpp_duktape_exec_timeout_check(udata)

#include "custom_sampler_duktape_runtime.h"
#include "vendor/duktape/duktape.c"
