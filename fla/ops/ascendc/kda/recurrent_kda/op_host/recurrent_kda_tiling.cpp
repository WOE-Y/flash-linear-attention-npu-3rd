/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_tiling.cpp
 * \brief
 */
#include "recurrent_kda_tiling.h"

#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus RecurrentKdaTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentKda", "context is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
        return Tiling4RecurrentKdaArch35(context);
    }
    return Tiling4RecurrentKdaArch22(context);
}

static ge::graphStatus TilingPrepareForRecurrentKda(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentKda", "context is null"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentKdaCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentKda)
    .Tiling(RecurrentKdaTilingFunc)
    .TilingParse<RecurrentKdaCompileInfo>(TilingPrepareForRecurrentKda);
} // namespace optiling
