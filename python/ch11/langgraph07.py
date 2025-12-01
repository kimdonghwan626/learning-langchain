from fastmcp import FastMCP, Context
from typing import Literal, Optional, Union, Dict

mcp = FastMCP("Unit Converter")

# ---------------------------------------------------------
# 온도 단위 변환 툴
# ---------------------------------------------------------
@mcp.tool()
def convert_temperature(
    value: float,
    source_unit: Literal["celsius", "fahrenheit"],
    target_unit: Literal["celsius", "fahrenheit"],
) -> Dict[str, Union[float, str]]:
    """
    온도 단위 변환 함수

    Parameters:
        value: 변환할 온도 값
        source_unit: 원본 단위 ('celsius' 또는 'fahrenheit')
        target_unit: 목표 단위 ('celsius' 또는 'fahrenheit')

    Returns:
        변환된 온도 값과 단위 정보를 포함하는 딕셔너리
    """

    if source_unit == target_unit:
        result = value
    elif source_unit == 'celsius' and target_unit == 'fahrenheit':
        result = (value * 9/5) + 32
    elif source_unit == 'fahrenheit' and target_unit == 'celsius':
        result = (value - 32) * 5/9
    else:
        raise ValueError("Unsupported temperature conversion")

    return {
        "value": result,
        "unit": target_unit,
        "original_value": value,
        "original_unit": source_unit
    }


# ---------------------------------------------------------
# 길이 단위 변환 툴
# ---------------------------------------------------------
@mcp.tool()
def convert_length(
    value: float,
    source_unit: Literal["meter", "feet", "inch", "cm"],
    target_unit: Literal["meter", "feet", "inch", "cm"],
) -> Dict[str, Union[float, str]]:
    """
    길이 단위 변환 함수

    Parameters:
        value: 변환할 길이 값
        source_unit: 원본 단위 ('meter', 'feet', 'inch', 'cm')
        target_unit: 목표 단위 ('meter', 'feet', 'inch', 'cm')

    Returns:
        변환된 길이 값과 단위 정보를 포함하는 딕셔너리
    """

    # 모든 단위를 미터로 변환하는 계수
    conversion_to_meter = {
        'meter': 1,
        'feet': 0.3048,
        'inch': 0.0254,
        'cm': 0.01,
    }

    if source_unit == target_unit:
        result = value
    else:
        # 원본 단위를 미터로 변환 후 목표 단위로 변환
        meter_value = value * conversion_to_meter[source_unit]
        result = meter_value / conversion_to_meter[target_unit]

    return {
        "value": result,
        "unit": target_unit,
        "original_value": value,
        "original_unit": source_unit
    }


# ---------------------------------------------------------
# 무게 단위 변환 툴
# ---------------------------------------------------------
@mcp.tool()
def convert_weight(
    value: float,
    source_unit: Literal["kg", "pound", "gram", "ounce"],
    target_unit: Literal["kg", "pound", "gram", "ounce"],
) -> Dict[str, Union[float, str]]:
    """
    무게 단위 변환 함수

    Parameters:
        value: 변환할 무게 값
        source_unit: 원본 단위 ('kg', 'pound', 'gram', 'ounce')
        target_unit: 목표 단위 ('kg', 'pound', 'gram', 'ounce')

    Returns:
        변환된 무게 값과 단위 정보를 포함하는 딕셔너리
    """

    # 모든 단위를 킬로그램으로 변환하는 계수
    conversion_to_kg = {
        'kg': 1,
        'pound': 0.45359237,
        'gram': 0.001,
        'ounce': 0.02834952,
    }

    if source_unit == target_unit:
        result = value
    else:
        # 원본 단위를 kg로 변환 후 목표 단위로 변환
        kg_value = value * conversion_to_kg[source_unit]
        result = kg_value / conversion_to_kg[target_unit]

    return {
        "value": result,
        "unit": target_unit,
        "original_value": value,
        "original_unit": source_unit
    }


# ---------------------------------------------------------
# MCP 서버 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    mcp.run()