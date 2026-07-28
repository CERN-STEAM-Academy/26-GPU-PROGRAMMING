#pragma once

#include <thrust/execution_policy.h>

struct multi_gpu_tag : thrust::execution_policy<multi_gpu_tag> {};

#include "copy.h"
#include "binary_search.h"
#include "for_each.h"
#include "gather.h"
#include "merge.h"
#include "remove.h"
#include "reverse.h"
#include "scan.h"
#include "scatter.h"
#include "sort.h"
#include "transform.h"
#include "unique.h"

static const multi_gpu_tag multi_gpu_policy;
