struct PS_INPUT {
    [[vk::location(0)]]
    float4 Color : COLOR0;
    [[vk::location(1)]]
    float2 UV : UV0;
};

[[vk::combinedImageSampler, vk::binding(0, 0)]]
Texture2D<float4> font_texture;

[[vk::combinedImageSampler, vk::binding(0, 0)]]
SamplerState font_sampler;

float4 main(in PS_INPUT input) : SV_TARGET {
    return input.Color * font_texture.Sample(font_sampler, input.UV);
}