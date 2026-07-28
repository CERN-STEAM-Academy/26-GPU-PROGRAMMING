#pragma once

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Multicore CPU: Execution Policy
#include "detail/execution_policy.h"

// Multicore CPU: Function Overloads
#include "detail/sort.h"
#include "detail/unique.h"
#include "detail/merge.h"
#include "detail/binary_search.h"
#include "detail/transform.h"
#include "detail/scan.h"
#include "detail/scatter.h"
#include "detail/gather.h"
#include "detail/for_each.h"
#include "detail/copy.h"
#include "detail/remove.h"
#include "detail/reverse.h"
