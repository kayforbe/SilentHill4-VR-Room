#define _CRT_SECURE_NO_WARNINGS
#define XR_USE_GRAPHICS_API_D3D11
#define XR_USE_PLATFORM_WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <DirectXMath.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <map>
#include <cmath>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

using namespace DirectX;

// ============ CONFIG ============
#define ROOM_PATH   "room.glb"
#define ROOM_SCALE  0.005f    // كبّر/صغّر الغرفة إذا جات بحجم غلط
#define MOVE_SPEED  1.5f       // متر / ثانية

#define XR_CHECK(expr) do { \
    XrResult _r = (expr); \
    if (XR_FAILED(_r)) { std::cerr << "XR err " << _r << " @ " #expr << "\n"; return 1; } \
} while(0)

// ============ State ============
XrInstance g_inst = XR_NULL_HANDLE;
XrSystemId g_sys = XR_NULL_SYSTEM_ID;
XrSession  g_ses = XR_NULL_HANDLE;
XrSpace    g_app = XR_NULL_HANDLE;
ID3D11Device* g_dev = nullptr;
ID3D11DeviceContext* g_ctx = nullptr;
XrSessionState g_state = XR_SESSION_STATE_UNKNOWN;
bool g_running = false;

XrActionSet g_actSet = XR_NULL_HANDLE;
XrAction    g_actMove = XR_NULL_HANDLE;
XrAction    g_actTurn = XR_NULL_HANDLE;
XrPath      g_leftPath;
XrPath      g_rightPath;

struct Swapchain {
    XrSwapchain handle; int32_t w, h;
    std::vector<XrSwapchainImageD3D11KHR> images;
    std::vector<ID3D11RenderTargetView*>  rtvs;
    std::vector<ID3D11DepthStencilView*>  dsvs;
    std::vector<ID3D11Texture2D*>         depthTex;
};
std::vector<Swapchain> g_sw;
std::vector<XrViewConfigurationView> g_vcfg;
std::vector<XrView> g_views;

ID3D11VertexShader* g_vs = nullptr;
ID3D11PixelShader* g_ps = nullptr;
ID3D11InputLayout* g_layout = nullptr;
ID3D11Buffer* g_cb = nullptr;
ID3D11RasterizerState* g_rs = nullptr;
ID3D11DepthStencilState* g_ds = nullptr;
ID3D11SamplerState* g_samp = nullptr;
ID3D11ShaderResourceView* g_whiteTex = nullptr;

struct Vertex { float x, y, z; float u, v; };
struct CB { XMFLOAT4X4 mvp; };

struct Submesh {
    ID3D11Buffer* vb = nullptr;
    ID3D11Buffer* ib = nullptr;
    UINT indexCount = 0;
    ID3D11ShaderResourceView* tex = nullptr;
    XMFLOAT4X4 world;
};
std::vector<Submesh> g_subs;
std::map<cgltf_image*, ID3D11ShaderResourceView*> g_texCache;

// مكان بداية اللاعب الذي طلبته
XMFLOAT3 g_playerPos = { 0, 0, 1.0f };
float g_playerYaw = 0.0f; // زاوية دوران اللاعب

// ============ Shader ============
const char* kHLSL = R"(
cbuffer CB : register(b0) { float4x4 mvp; };
Texture2D tex : register(t0);
SamplerState samp : register(s0);
struct VSIn  { float3 pos : POSITION; float2 uv : TEXCOORD; };
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };
VSOut VSMain(VSIn i) {
    VSOut o;
    o.pos = mul(float4(i.pos, 1.0), mvp);
    o.uv = i.uv;
    return o;
}
float4 PSMain(VSOut i) : SV_TARGET {
    return tex.Sample(samp, i.uv);
}
)";

XMMATRIX ProjFromFov(const XrFovf& fov, float nz, float fz) {
    float l = std::tan(fov.angleLeft);
    float r = std::tan(fov.angleRight);
    float u = std::tan(fov.angleUp);
    float d = std::tan(fov.angleDown);
    float w = r - l, h = u - d;
    float zr = fz / (nz - fz);
    XMFLOAT4X4 m = {
        2 / w,   0,     0,       0,
        0,     2 / h,   0,       0,
        (r + l) / w,(u + d) / h,zr,   -1,
        0,     0,     nz * zr,   0
    };
    return XMLoadFloat4x4(&m);
}

// ============ glTF loading ============
XMMATRIX NodeLocal(const cgltf_node* n) {
    if (n->has_matrix) {
        XMFLOAT4X4 m;
        for (int i = 0; i < 16; i++) ((float*)&m)[i] = n->matrix[i];
        return XMLoadFloat4x4(&m);
    }
    XMMATRIX T = n->has_translation
        ? XMMatrixTranslation(n->translation[0], n->translation[1], n->translation[2])
        : XMMatrixIdentity();
    XMMATRIX R = n->has_rotation
        ? XMMatrixRotationQuaternion(XMVectorSet(n->rotation[0], n->rotation[1],
            n->rotation[2], n->rotation[3]))
        : XMMatrixIdentity();
    XMMATRIX S = n->has_scale
        ? XMMatrixScaling(n->scale[0], n->scale[1], n->scale[2])
        : XMMatrixIdentity();
    return S * R * T;
}

ID3D11ShaderResourceView* UploadPixels(const uint8_t* data, size_t size) {
    int w, h, n;
    unsigned char* px = stbi_load_from_memory(data, (int)size, &w, &h, &n, 4);
    if (!px) return nullptr;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = px; sd.SysMemPitch = w * 4;
    ID3D11Texture2D* tex = nullptr;
    g_dev->CreateTexture2D(&td, &sd, &tex);
    stbi_image_free(px);
    if (!tex) return nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    g_dev->CreateShaderResourceView(tex, nullptr, &srv);
    tex->Release();
    return srv;
}

ID3D11ShaderResourceView* GetTexture(cgltf_image* img) {
    if (!img) return g_whiteTex;
    auto it = g_texCache.find(img);
    if (it != g_texCache.end()) return it->second;
    ID3D11ShaderResourceView* srv = nullptr;
    if (img->buffer_view) {
        const uint8_t* base = (const uint8_t*)img->buffer_view->buffer->data;
        srv = UploadPixels(base + img->buffer_view->offset, img->buffer_view->size);
    }
    if (!srv) srv = g_whiteTex;
    g_texCache[img] = srv;
    return srv;
}

void ProcessPrimitive(const cgltf_primitive* p, XMMATRIX world) {
    if (p->type != cgltf_primitive_type_triangles || !p->indices) return;
    const cgltf_accessor* posA = nullptr;
    const cgltf_accessor* uvA = nullptr;
    for (cgltf_size i = 0; i < p->attributes_count; i++) {
        const auto& a = p->attributes[i];
        if (a.type == cgltf_attribute_type_position) posA = a.data;
        else if (a.type == cgltf_attribute_type_texcoord && a.index == 0) uvA = a.data;
    }
    if (!posA) return;

    cgltf_size vc = posA->count;
    std::vector<float> pos(vc * 3);
    cgltf_accessor_unpack_floats(posA, pos.data(), vc * 3);
    std::vector<float> uvs(vc * 2, 0.0f);
    if (uvA && uvA->count == vc)
        cgltf_accessor_unpack_floats(uvA, uvs.data(), vc * 2);

    std::vector<Vertex> V(vc);
    for (cgltf_size i = 0; i < vc; i++) {
        V[i].x = pos[i * 3 + 0]; V[i].y = pos[i * 3 + 1]; V[i].z = pos[i * 3 + 2];
        V[i].u = uvs[i * 2 + 0]; V[i].v = uvs[i * 2 + 1];
    }
    cgltf_size ic = p->indices->count;
    std::vector<uint32_t> I(ic);
    for (cgltf_size i = 0; i < ic; i++)
        I[i] = (uint32_t)cgltf_accessor_read_index(p->indices, i);

    Submesh s;
    XMStoreFloat4x4(&s.world, world);
    s.indexCount = (UINT)ic;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = (UINT)(V.size() * sizeof(Vertex));
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA sd{ V.data() };
    g_dev->CreateBuffer(&bd, &sd, &s.vb);
    bd.ByteWidth = (UINT)(I.size() * sizeof(uint32_t));
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    sd.pSysMem = I.data();
    g_dev->CreateBuffer(&bd, &sd, &s.ib);

    cgltf_image* img = nullptr;
    if (p->material && p->material->has_pbr_metallic_roughness) {
        auto& pbr = p->material->pbr_metallic_roughness;
        if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image)
            img = pbr.base_color_texture.texture->image;
    }
    s.tex = GetTexture(img);
    g_subs.push_back(s);
}

void ProcessNode(const cgltf_node* n, XMMATRIX parent) {
    XMMATRIX world = NodeLocal(n) * parent;
    if (n->mesh)
        for (cgltf_size i = 0; i < n->mesh->primitives_count; i++)
            ProcessPrimitive(&n->mesh->primitives[i], world);
    for (cgltf_size i = 0; i < n->children_count; i++)
        ProcessNode(n->children[i], world);
}

bool LoadGLB(const char* path) {
    cgltf_options opts{};
    cgltf_data* d = nullptr;
    if (cgltf_parse_file(&opts, path, &d) != cgltf_result_success) {
        std::cerr << "[FAIL] parse: " << path << "\n"; return false;
    }
    if (cgltf_load_buffers(&opts, d, path) != cgltf_result_success) {
        std::cerr << "[FAIL] load buffers\n"; cgltf_free(d); return false;
    }
    XMMATRIX root = XMMatrixScaling(ROOM_SCALE, ROOM_SCALE, ROOM_SCALE);
    cgltf_scene* sc = d->scene ? d->scene : (d->scenes_count ? &d->scenes[0] : nullptr);
    if (!sc) { std::cerr << "[FAIL] no scene\n"; cgltf_free(d); return false; }
    for (cgltf_size i = 0; i < sc->nodes_count; i++) ProcessNode(sc->nodes[i], root);
    std::cout << "[OK] Loaded " << g_subs.size() << " submeshes, "
        << g_texCache.size() << " textures\n";
    cgltf_free(d);
    return true;
}

int main() {
    std::cout << "=== Stage 5: Walking in room.glb ===\n";

    // Instance
    {
        const char* ex[] = { XR_KHR_D3D11_ENABLE_EXTENSION_NAME };
        XrInstanceCreateInfo ci{ XR_TYPE_INSTANCE_CREATE_INFO };
        ci.enabledExtensionCount = 1; ci.enabledExtensionNames = ex;
        strcpy_s(ci.applicationInfo.applicationName, "MyVRApp");
        ci.applicationInfo.applicationVersion = 1;
        strcpy_s(ci.applicationInfo.engineName, "None");
        ci.applicationInfo.engineVersion = 1;
        ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        XR_CHECK(xrCreateInstance(&ci, &g_inst));
    }
    // System
    {
        XrSystemGetInfo gi{ XR_TYPE_SYSTEM_GET_INFO };
        gi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XR_CHECK(xrGetSystem(g_inst, &gi, &g_sys));
    }
    // D3D11 device
    LUID luid{};
    {
        PFN_xrGetD3D11GraphicsRequirementsKHR pfn = nullptr;
        XR_CHECK(xrGetInstanceProcAddr(g_inst, "xrGetD3D11GraphicsRequirementsKHR",
            (PFN_xrVoidFunction*)&pfn));
        XrGraphicsRequirementsD3D11KHR req{ XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
        XR_CHECK(pfn(g_inst, g_sys, &req));
        luid = req.adapterLuid;
    }
    {
        IDXGIFactory1* f = nullptr;
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&f);
        IDXGIAdapter1* a = nullptr;
        for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
            if (memcmp(&d.AdapterLuid, &luid, sizeof(LUID)) == 0) break;
            a->Release(); a = nullptr;
        }
        f->Release();
        D3D_FEATURE_LEVEL lvl[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        D3D_FEATURE_LEVEL got;
        D3D11CreateDevice(a, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, lvl, 2,
            D3D11_SDK_VERSION, &g_dev, &got, &g_ctx);
        a->Release();
    }
    // Session
    {
        XrGraphicsBindingD3D11KHR bind{ XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };
        bind.device = g_dev;
        XrSessionCreateInfo sci{ XR_TYPE_SESSION_CREATE_INFO };
        sci.next = &bind; sci.systemId = g_sys;
        XR_CHECK(xrCreateSession(g_inst, &sci, &g_ses));
    }
    // Reference space: STAGE (floor) preferred
    {
        XrReferenceSpaceCreateInfo rsi{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        rsi.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        rsi.poseInReferenceSpace.orientation.w = 1.0f;
        if (XR_FAILED(xrCreateReferenceSpace(g_ses, &rsi, &g_app))) {
            rsi.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            XR_CHECK(xrCreateReferenceSpace(g_ses, &rsi, &g_app));
            std::cout << "[*] Using LOCAL (no floor)\n";
        }
        else std::cout << "[*] Using STAGE (floor)\n";
    }
    // Actions
    {
        XrActionSetCreateInfo asci{ XR_TYPE_ACTION_SET_CREATE_INFO };
        strcpy_s(asci.actionSetName, "gameplay");
        strcpy_s(asci.localizedActionSetName, "Gameplay");
        XR_CHECK(xrCreateActionSet(g_inst, &asci, &g_actSet));

        // إعداد عصا المشي (يسار)
        xrStringToPath(g_inst, "/user/hand/left", &g_leftPath);
        XrActionCreateInfo aciMove{ XR_TYPE_ACTION_CREATE_INFO };
        aciMove.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
        strcpy_s(aciMove.actionName, "move");
        strcpy_s(aciMove.localizedActionName, "Move");
        aciMove.countSubactionPaths = 1;
        aciMove.subactionPaths = &g_leftPath;
        XR_CHECK(xrCreateAction(g_actSet, &aciMove, &g_actMove));

        // إعداد عصا الالتفاف (يمين)
        xrStringToPath(g_inst, "/user/hand/right", &g_rightPath);
        XrActionCreateInfo aciTurn{ XR_TYPE_ACTION_CREATE_INFO };
        aciTurn.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
        strcpy_s(aciTurn.actionName, "turn");
        strcpy_s(aciTurn.localizedActionName, "Turn");
        aciTurn.countSubactionPaths = 1;
        aciTurn.subactionPaths = &g_rightPath;
        XR_CHECK(xrCreateAction(g_actSet, &aciTurn, &g_actTurn));

        // ربط الأزرار
        XrPath leftStick, rightStick, profile;
        xrStringToPath(g_inst, "/user/hand/left/input/thumbstick", &leftStick);
        xrStringToPath(g_inst, "/user/hand/right/input/thumbstick", &rightStick);
        xrStringToPath(g_inst, "/interaction_profiles/oculus/touch_controller", &profile);

        XrActionSuggestedBinding bindings[2] = {
            { g_actMove, leftStick },
            { g_actTurn, rightStick }
        };
        XrInteractionProfileSuggestedBinding sug{ XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
        sug.interactionProfile = profile;
        sug.countSuggestedBindings = 2;
        sug.suggestedBindings = bindings;
        XR_CHECK(xrSuggestInteractionProfileBindings(g_inst, &sug));

        XrSessionActionSetsAttachInfo att{ XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
        att.countActionSets = 1; att.actionSets = &g_actSet;
        XR_CHECK(xrAttachSessionActionSets(g_ses, &att));
    }
    // Views
    {
        uint32_t n = 0;
        xrEnumerateViewConfigurationViews(g_inst, g_sys,
            XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &n, nullptr);
        g_vcfg.resize(n, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
        xrEnumerateViewConfigurationViews(g_inst, g_sys,
            XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, n, &n, g_vcfg.data());
        g_views.resize(n, { XR_TYPE_VIEW });
    }
    // Swapchains
    DXGI_FORMAT colorFmt = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    {
        uint32_t fc = 0;
        xrEnumerateSwapchainFormats(g_ses, 0, &fc, nullptr);
        std::vector<int64_t> fmts(fc);
        xrEnumerateSwapchainFormats(g_ses, fc, &fc, fmts.data());
        for (auto f : fmts)
            if (f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)
            {
                colorFmt = (DXGI_FORMAT)f; break;
            }

        g_sw.resize(g_vcfg.size());
        for (size_t i = 0; i < g_vcfg.size(); i++) {
            auto& vc = g_vcfg[i];
            XrSwapchainCreateInfo sci{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
            sci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
            sci.format = colorFmt; sci.sampleCount = 1;
            sci.width = vc.recommendedImageRectWidth;
            sci.height = vc.recommendedImageRectHeight;
            sci.faceCount = 1; sci.arraySize = 1; sci.mipCount = 1;
            xrCreateSwapchain(g_ses, &sci, &g_sw[i].handle);
            g_sw[i].w = sci.width; g_sw[i].h = sci.height;

            uint32_t ic = 0;
            xrEnumerateSwapchainImages(g_sw[i].handle, 0, &ic, nullptr);
            g_sw[i].images.resize(ic, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
            xrEnumerateSwapchainImages(g_sw[i].handle, ic, &ic,
                (XrSwapchainImageBaseHeader*)g_sw[i].images.data());
            g_sw[i].rtvs.resize(ic); g_sw[i].dsvs.resize(ic); g_sw[i].depthTex.resize(ic);
            for (uint32_t j = 0; j < ic; j++) {
                D3D11_RENDER_TARGET_VIEW_DESC rd{};
                rd.Format = colorFmt;
                rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
                g_dev->CreateRenderTargetView(g_sw[i].images[j].texture, &rd, &g_sw[i].rtvs[j]);
                D3D11_TEXTURE2D_DESC td{};
                td.Width = sci.width; td.Height = sci.height;
                td.MipLevels = 1; td.ArraySize = 1;
                td.Format = DXGI_FORMAT_D32_FLOAT;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
                g_dev->CreateTexture2D(&td, nullptr, &g_sw[i].depthTex[j]);
                g_dev->CreateDepthStencilView(g_sw[i].depthTex[j], nullptr, &g_sw[i].dsvs[j]);
            }
        }
    }
    // Shaders
    {
        ID3DBlob* vsb = nullptr, * psb = nullptr, * err = nullptr;
        D3DCompile(kHLSL, strlen(kHLSL), nullptr, nullptr, nullptr,
            "VSMain", "vs_5_0", 0, 0, &vsb, &err);
        if (err) { std::cerr << (char*)err->GetBufferPointer(); return 1; }
        D3DCompile(kHLSL, strlen(kHLSL), nullptr, nullptr, nullptr,
            "PSMain", "ps_5_0", 0, 0, &psb, &err);
        if (err) { std::cerr << (char*)err->GetBufferPointer(); return 1; }
        g_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_vs);
        g_dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_ps);
        D3D11_INPUT_ELEMENT_DESC ied[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        g_dev->CreateInputLayout(ied, 2, vsb->GetBufferPointer(), vsb->GetBufferSize(), &g_layout);
        vsb->Release(); psb->Release();
    }
    // CB, rasterizer, depth, sampler, white tex
    {
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = sizeof(CB);
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        g_dev->CreateBuffer(&bd, nullptr, &g_cb);

        D3D11_RASTERIZER_DESC rd{};
        rd.FillMode = D3D11_FILL_SOLID;
        rd.CullMode = D3D11_CULL_NONE;     // مهم: ما نقطع أي وجه
        rd.FrontCounterClockwise = TRUE;
        g_dev->CreateRasterizerState(&rd, &g_rs);

        D3D11_DEPTH_STENCIL_DESC dd{};
        dd.DepthEnable = TRUE;
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dd.DepthFunc = D3D11_COMPARISON_LESS;
        g_dev->CreateDepthStencilState(&dd, &g_ds);

        D3D11_SAMPLER_DESC sd2{};
        sd2.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd2.AddressU = sd2.AddressV = sd2.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sd2.MaxLOD = D3D11_FLOAT32_MAX;
        g_dev->CreateSamplerState(&sd2, &g_samp);

        uint32_t white = 0xFFFFFFFF;
        D3D11_TEXTURE2D_DESC td{};
        td.Width = 1; td.Height = 1; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = &white; sd.SysMemPitch = 4;
        ID3D11Texture2D* tex = nullptr;
        g_dev->CreateTexture2D(&td, &sd, &tex);
        g_dev->CreateShaderResourceView(tex, nullptr, &g_whiteTex);
        tex->Release();
    }

    if (!LoadGLB(ROOM_PATH)) return 1;

    std::cout << "[*] Put on headset. LEFT thumbstick = walk. RIGHT thumbstick = turn.\n";

    // Main loop
    bool quit = false;
    while (!quit) {
        XrEventDataBuffer ev{ XR_TYPE_EVENT_DATA_BUFFER };
        while (xrPollEvent(g_inst, &ev) == XR_SUCCESS) {
            if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                auto* s = (XrEventDataSessionStateChanged*)&ev;
                g_state = s->state;
                if (g_state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo bi{ XR_TYPE_SESSION_BEGIN_INFO };
                    bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    xrBeginSession(g_ses, &bi); g_running = true;
                }
                else if (g_state == XR_SESSION_STATE_STOPPING) {
                    xrEndSession(g_ses); g_running = false;
                }
                else if (g_state == XR_SESSION_STATE_EXITING ||
                    g_state == XR_SESSION_STATE_LOSS_PENDING) {
                    quit = true;
                }
            }
            ev = { XR_TYPE_EVENT_DATA_BUFFER };
        }
        if (!g_running) { Sleep(50); continue; }

        XrFrameState fs{ XR_TYPE_FRAME_STATE };
        XrFrameWaitInfo fwi{ XR_TYPE_FRAME_WAIT_INFO };
        xrWaitFrame(g_ses, &fwi, &fs);
        XrFrameBeginInfo fbi{ XR_TYPE_FRAME_BEGIN_INFO };
        xrBeginFrame(g_ses, &fbi);

        // sync input
        XrActiveActionSet aas{ g_actSet, XR_NULL_PATH };
        XrActionsSyncInfo si{ XR_TYPE_ACTIONS_SYNC_INFO };
        si.countActiveActionSets = 1; si.activeActionSets = &aas;
        xrSyncActions(g_ses, &si);

        XrActionStateVector2f stick{ XR_TYPE_ACTION_STATE_VECTOR2F };
        {
            XrActionStateGetInfo gi{ XR_TYPE_ACTION_STATE_GET_INFO };
            gi.action = g_actMove; gi.subactionPath = g_leftPath;
            xrGetActionStateVector2f(g_ses, &gi, &stick);
        }

        XrActionStateVector2f stickTurn{ XR_TYPE_ACTION_STATE_VECTOR2F };
        {
            XrActionStateGetInfo gi{ XR_TYPE_ACTION_STATE_GET_INFO };
            gi.action = g_actTurn; gi.subactionPath = g_rightPath;
            xrGetActionStateVector2f(g_ses, &gi, &stickTurn);
        }

        std::vector<XrCompositionLayerProjectionView> pv;
        XrCompositionLayerProjection layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };

        if (fs.shouldRender) {
            XrViewState vs{ XR_TYPE_VIEW_STATE };
            XrViewLocateInfo vli{ XR_TYPE_VIEW_LOCATE_INFO };
            vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            vli.displayTime = fs.predictedDisplayTime;
            vli.space = g_app;
            uint32_t vc = 0;
            xrLocateViews(g_ses, &vli, &vs, (uint32_t)g_views.size(), &vc, g_views.data());

            // --- locomotion ---
            float dt = (float)(fs.predictedDisplayPeriod) / 1.0e9f;

            // حساب الالتفاف (عصا يمين) - تم إصلاح مشكلة دوران البندول هنا!
            if (stickTurn.isActive && stickTurn.currentState.x != 0) {
                float deltaYaw = stickTurn.currentState.x * 2.5f * dt;

                // قراءة مكان النظارة الحقيقي (الرأس) عشان ندور حوله
                float hx = g_views[0].pose.position.x;
                float hz = g_views[0].pose.position.z;

                // حساب موقع الرأس قبل الدوران
                float old_offsetX = hx * cosf(g_playerYaw) + hz * sinf(g_playerYaw);
                float old_offsetZ = -hx * sinf(g_playerYaw) + hz * cosf(g_playerYaw);

                g_playerYaw += deltaYaw; // تطبيق الدوران

                // حساب موقع الرأس بعد الدوران
                float new_offsetX = hx * cosf(g_playerYaw) + hz * sinf(g_playerYaw);
                float new_offsetZ = -hx * sinf(g_playerYaw) + hz * cosf(g_playerYaw);

                // نعكس الإزاحة عشان اللاعب ما يطير كأنه في دائرة ويبقى في مكانه
                g_playerPos.x += (old_offsetX - new_offsetX);
                g_playerPos.z += (old_offsetZ - new_offsetZ);
            }

            // حساب المشي (عصا يسار)
            if (stick.isActive && (stick.currentState.x != 0 || stick.currentState.y != 0)) {
                const XrQuaternionf& q = g_views[0].pose.orientation;
                XMVECTOR qv = XMVectorSet(q.x, q.y, q.z, q.w);
                XMVECTOR fwd = XMVector3Rotate(XMVectorSet(0, 0, -1, 0), qv);
                fwd = XMVectorSetY(fwd, 0);
                fwd = XMVector3Normalize(fwd);
                XMVECTOR rightv = XMVectorSet(-XMVectorGetZ(fwd), 0, XMVectorGetX(fwd), 0);

                XMMATRIX rotY = XMMatrixRotationY(g_playerYaw);
                fwd = XMVector3TransformNormal(fwd, rotY);
                rightv = XMVector3TransformNormal(rightv, rotY);

                XMVECTOR p = XMLoadFloat3(&g_playerPos);
                p = XMVectorAdd(p, XMVectorScale(fwd, stick.currentState.y * MOVE_SPEED * dt));
                p = XMVectorAdd(p, XMVectorScale(rightv, stick.currentState.x * MOVE_SPEED * dt));
                XMStoreFloat3(&g_playerPos, p);
            }

            pv.resize(vc);
            for (uint32_t i = 0; i < vc; i++) {
                auto& s = g_sw[i];
                uint32_t ix = 0;
                XrSwapchainImageAcquireInfo ai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
                xrAcquireSwapchainImage(s.handle, &ai, &ix);
                XrSwapchainImageWaitInfo wi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
                wi.timeout = XR_INFINITE_DURATION;
                xrWaitSwapchainImage(s.handle, &wi);

                float sky[4] = { 0.1f, 0.12f, 0.15f, 1 };
                g_ctx->ClearRenderTargetView(s.rtvs[ix], sky);
                g_ctx->ClearDepthStencilView(s.dsvs[ix], D3D11_CLEAR_DEPTH, 1.0f, 0);
                g_ctx->OMSetRenderTargets(1, &s.rtvs[ix], s.dsvs[ix]);
                D3D11_VIEWPORT vp{ 0, 0, (float)s.w, (float)s.h, 0, 1 };
                g_ctx->RSSetViewports(1, &vp);
                g_ctx->RSSetState(g_rs);
                g_ctx->OMSetDepthStencilState(g_ds, 0);

                const XrPosef& p = g_views[i].pose;

                // حساب الكاميرا مع الدوران
                XMVECTOR headQ = XMVectorSet(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
                XMVECTOR headPos = XMVectorSet(p.position.x, p.position.y, p.position.z, 0);

                XMMATRIX headRotMat = XMMatrixRotationQuaternion(headQ);
                XMMATRIX headTransMat = XMMatrixTranslationFromVector(headPos);

                XMMATRIX playerRotMat = XMMatrixRotationY(g_playerYaw);
                XMMATRIX playerTransMat = XMMatrixTranslation(g_playerPos.x, g_playerPos.y, g_playerPos.z);

                XMMATRIX camWorld = headRotMat * headTransMat * playerRotMat * playerTransMat;
                XMMATRIX view = XMMatrixInverse(nullptr, camWorld);

                XMMATRIX proj = ProjFromFov(g_views[i].fov, 0.05f, 500.0f);
                XMMATRIX viewProj = view * proj;

                g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                g_ctx->IASetInputLayout(g_layout);
                g_ctx->VSSetShader(g_vs, nullptr, 0);
                g_ctx->VSSetConstantBuffers(0, 1, &g_cb);
                g_ctx->PSSetShader(g_ps, nullptr, 0);
                g_ctx->PSSetSamplers(0, 1, &g_samp);

                UINT stride = sizeof(Vertex), off = 0;
                for (auto& sub : g_subs) {
                    XMMATRIX world = XMLoadFloat4x4(&sub.world);
                    XMMATRIX mvp = XMMatrixTranspose(world * viewProj);
                    D3D11_MAPPED_SUBRESOURCE m{};
                    g_ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
                    XMStoreFloat4x4((XMFLOAT4X4*)m.pData, mvp);
                    g_ctx->Unmap(g_cb, 0);
                    g_ctx->IASetVertexBuffers(0, 1, &sub.vb, &stride, &off);
                    g_ctx->IASetIndexBuffer(sub.ib, DXGI_FORMAT_R32_UINT, 0);
                    g_ctx->PSSetShaderResources(0, 1, &sub.tex);
                    g_ctx->DrawIndexed(sub.indexCount, 0, 0);
                }

                XrSwapchainImageReleaseInfo ri{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
                xrReleaseSwapchainImage(s.handle, &ri);

                pv[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
                pv[i].pose = g_views[i].pose;
                pv[i].fov = g_views[i].fov;
                pv[i].subImage.swapchain = s.handle;
                pv[i].subImage.imageRect.offset = { 0, 0 };
                pv[i].subImage.imageRect.extent = { s.w, s.h };
            }
            layer.space = g_app;
            layer.viewCount = (uint32_t)pv.size();
            layer.views = pv.data();
        }

        XrCompositionLayerBaseHeader* ls[] = { (XrCompositionLayerBaseHeader*)&layer };
        XrFrameEndInfo fei{ XR_TYPE_FRAME_END_INFO };
        fei.displayTime = fs.predictedDisplayTime;
        fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        fei.layerCount = fs.shouldRender ? 1u : 0u;
        fei.layers = fs.shouldRender ? ls : nullptr;
        xrEndFrame(g_ses, &fei);
    }
    return 0;
}