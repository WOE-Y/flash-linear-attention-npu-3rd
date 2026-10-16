/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_tiling.h
 * \brief
 */
#ifndef OP_HOST_RECURRENT_KDA_TILING_H
#define OP_HOST_RECURRENT_KDA_TILING_H

#include "register/op_impl_registry.h"

namespace optiling {

struct RecurrentKdaCompileInfo {};

ge::graphStatus Tiling4RecurrentKdaArch22(gert::TilingContext *context);
ge::graphStatus Tiling4RecurrentKdaArch35(gert::TilingContext *context);

} // namespace optiling

#endif // OP_HOST_RECURRENT_KDA_TILING_H
