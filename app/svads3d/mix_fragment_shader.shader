#version 330 core

in vec2 TexCoord;     // 从顶点着色器传递过来的纹理坐标
in float mixFactorA;  // 从顶点着色器传递过来的纹理A的混合因子
in float mixFactorB;  // 从顶点着色器传递过来的纹理B的混合因子

out vec4 FragColor;   // 输出的颜色

uniform sampler2D textureA; // 纹理A
uniform sampler2D textureB; // 纹理B

void main()
{
    vec4 colorA = texture(textureA, TexCoord);
    vec4 colorB = texture(textureB, TexCoord);
    
    // 根据混合因子进行线性插值
    FragColor = mix(colorA, colorB, mixFactorA / (mixFactorA + mixFactorB));
}

