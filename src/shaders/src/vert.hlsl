struct VS_INPUT {
    [[vk::location(0)]]
    float2 Position : POSITION0;
    [[vk::location(1)]]
    float2 UV : UV0;
    [[vk::location(2)]]
    float4 Color : COLOR0;
};

struct VS_OUTPUT {
    float4 Position : SV_Position;
    [[vk::location(0)]]
    float4 Color : COLOR0;
    [[vk::location(1)]]
    float2 UV : UV0;
};

[[vk::push_constant]]
struct PushConstants {
    float2 screen_size;
} pc;

float3 srgb_to_linear(float3 srgb) {
    return srgb * (srgb * (srgb * 0.305306011 + 0.682171111) + 0.012522878);
}


VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.Position =
      float4(2.0 * input.Position.x / pc.screen_size.x - 1.0,
             2.0 * input.Position.y / pc.screen_size.y - 1.0,
             0.0,
             1.0);
    output.Color = float4(srgb_to_linear(input.Color.rgb), input.Color.a);
    output.UV = input.UV;
    return output;
}