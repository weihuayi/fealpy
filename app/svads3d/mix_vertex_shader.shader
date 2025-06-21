#version 330 core

layout(location = 0) in vec3 aPos;      // 位置变量
layout(location = 1) in vec2 aTexCoord; // 纹理坐标变量
layout(location = 2) in float weightA;  // 纹理A的权重
layout(location = 3) in float weightB;  // 纹理B的权重

out vec2 TexCoord;     // 传递给片段着色器的纹理坐标
out float mixFactorA;  // 传递给片段着色器的纹理A的混合因子
out float mixFactorB;  // 传递给片段着色器的纹理B的混合因子

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    mixFactorA = weightA;  // 将权重传递给片段着色器
    mixFactorB = weightB;  // 将权重传递给片段着色器
}

