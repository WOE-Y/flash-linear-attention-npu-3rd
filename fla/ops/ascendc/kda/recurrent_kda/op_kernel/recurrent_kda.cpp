/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/recurrent_kda_arch35.cpp"
#else
#include "arch22/recurrent_kda_arch22.cpp"
#endif
