// Copyright (c) 2025 hotflow2024
// Licensed under AGPL-3.0-or-later. See LICENSE for details.
// Commercial licensing available. See COMMERCIAL_LICENSE.md.

/**
 * Visual Style Presets - 视觉风格预设
 * 
 * 统一的视觉风格定义，所有板块（分镜、角色、场景、AI导演）共用
 */

// 风格分类
export type StyleCategory = 'real' | 'e_commerce' | 'street_snap' | 'commercial';

/**
 * 媒介类型 — 决定 prompt-builder 如何翻译摄影参数
 * - cinematic: 完整物理摄影词汇（真人/写实3D）
 * - graphic: 仅色彩/情绪/节奏（像素/水彩/简笔画等高度抽象风格）
 */
export type MediaType = 'cinematic' | 'graphic';

export interface StylePreset {
  id: string;
  name: string;
  category: StyleCategory;
  /** 媒介类型 — 控制摄影参数翻译策略 */
  mediaType: MediaType;
  /** 英文提示词 */
  prompt: string;
  /** 负面提示词 */
  negativePrompt: string;
  /** 中文描述 */
  description: string;
  /** 缩略图文件名 */
  thumbnail: string;
}

// ============================================================
// 真人/写实类 (Photography)
// ============================================================

const STYLES_REAL: StylePreset[] = [
  {
    id: 'real_movie',
    name: '真人电影',
    category: 'real',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, high detailed:1.2), (cinematic movie still:1.3), (35mm film grain:1.2), (dramatic movie lighting:1.1), (color graded:1.1), photorealistic, depth of field',
    negativePrompt: '(worst quality, low quality:1.4), (3D render, cgi, game), (anime, illustration, painting), (cartoon), artificial, fake',
    description: '电影剧照，胶片感，电影调色',
    thumbnail: 'real_movie.png',
  },
  {
    id: 'real_bloom',
    name: '真实光晕',
    category: 'real',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, high detailed:1.2), (dreamy soft focus photography:1.3), (strong bloom, lens flare:1.2), (backlit by sun:1.1), (ethereal lighting:1.1), photorealistic, angelic',
    negativePrompt: '(worst quality, low quality:1.4), (sharp, harsh contrast), (dark, gloomy), (anime, 3d), (flat lighting), ugly',
    description: '唯美光晕，逆光，梦幻光效',
    thumbnail: 'real_bloom.png',
  },
];

// ============================================================
// 电商垂直类 (Amazon/TikTok/Social)
// ============================================================

const STYLES_ECOMMERCE: StylePreset[] = [
  {
    id: 'amazon_model',
    name: 'Amazon 模特',
    category: 'e_commerce',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (fashion photography:1.2), (Amazon product catalog style:1.2), (full body shot:1.1), (clean white background:1.3), (even soft studio lighting:1.2), (highly detailed skin texture, realistic pores:1.1), sharp focus, natural colors, commercial quality',
    negativePrompt: '(worst quality, low quality:1.4), (2D, anime, cartoon, 3d render:1.3), (messy background, outdoor:1.2), (harsh shadows), (distorted face, bad anatomy:1.2), watermark, logo, text, blurry',
    description: '亚马逊电商风格，白底模特，摄影棚布光，写实肤质',
    thumbnail: 'amazon_model.png',
  },
  {
    id: 'buyer_show_natural',
    name: '写实买家秀',
    category: 'e_commerce',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (authentic buyer show photography:1.4), (handheld phone camera look, shot on iPhone:1.2), (natural indoor domestic lighting:1.2), (casual living room background:1.1), (unfiltered skin texture:1.1), (natural candid expression:1.2), non-professional lighting, realistic lifestyle, casual clothing',
    negativePrompt: '(worst quality, low quality:1.4), (studio lighting, professional photography:1.3), (perfect skin, airbrushed:1.2), (white background), (3d, anime, cgi), watermark, logo, fake',
    description: '真实买家秀，日常家居环境，手机抓拍质感，自然光影',
    thumbnail: 'buyer_show_natural.png',
  },
  {
    id: 'influencer_vlog',
    name: '达人演示',
    category: 'e_commerce',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (KOL/Influencer style vlogging still:1.3), (bright ring light reflection in eyes:1.2), (vlogger studio/bedroom setup:1.2), (colorful LED background lights:1.1), (enthusiastic smiling presentation:1.2), (modern fashionable lifestyle:1.1), sharp focus, professional desktop microphone visible in frame',
    negativePrompt: '(worst quality, low quality:1.4), (dark, gloomy), (low energy), (anime, 3d), watermark, signature, blurry',
    description: '达人/KOL视角的直播分享，环形补光，背景装饰丰富',
    thumbnail: 'influencer_vlog.png',
  },
  {
    id: 'tiktok_fashion',
    name: 'TikTok 潮流',
    category: 'e_commerce',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (TikTok style short video still:1.2), (dynamic energetic mood:1.1), (natural urban lighting:1.1), (handheld camera feel:1.1), trendy outfit, stylish posing, vibrant lifestyle, sharp focus, cinematic bokeh',
    negativePrompt: '(worst quality, low quality:1.4), (3d, anime), (studio background), (still, boring), watermark, text, signature',
    description: '短视频风格，动感活力，自然光，潮流穿搭',
    thumbnail: 'tiktok_fashion.png',
  },
  {
    id: 'product_close_up',
    name: '产品特写',
    category: 'e_commerce',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (macro photography:1.2), (product close-up:1.3), (studio lighting:1.1), (soft bokeh:1.1), extreme detail, clean background, sharp focus, ray tracing lighting',
    negativePrompt: '(worst quality, low quality:1.4), (person, face, model), (blurry, out of focus), (dirty background), text, low res',
    description: '产品细节拍摄，微距感，纯净背景',
    thumbnail: 'product_close_up.png',
  },
];

// ============================================================
// 广告/品牌类 (Brand/TVC)
// ============================================================

const STYLES_COMMERCIAL: StylePreset[] = [
  {
    id: 'product_tvc_ad',
    name: '产品TVC广告',
    category: 'commercial',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (high-end TVC commercial cinematography:1.4), (luxury lighting, dramatic contrast:1.3), (slow motion feel:1.1), (RED camera look:1.2), (brand color palette:1.1), (professional high-end post-production color grading:1.2), elegant composition, macro product details, sophisticated atmosphere',
    negativePrompt: '(worst quality, low quality:1.4), (cheap look, handheld, amateur:1.3), (messy, cluttered), (3d, anime), low resolution, blurry',
    description: '高端电视广告/TVC质感，电影级运镜光影，奢侈品级调色',
    thumbnail: 'product_tvc_ad.png',
  },
  {
    id: 'street_candid',
    name: '自然街拍',
    category: 'street_snap',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (candid street photography:1.3), (35mm lens:1.1), (urban street background:1.2), (natural daylight:1.1), (soft depth of field, bokeh:1.2), (authentic mood:1.1), sharp focus, walking character',
    negativePrompt: '(worst quality, low quality:1.4), (studio, white background), (posed, stiff), (3d, anime), low res, blurry',
    description: '单反街拍，自然抓拍，35mm镜头，街道背景',
    thumbnail: 'street_candid.png',
  },
  {
    id: 'cyberpunk_street',
    name: '都市赛博街拍',
    category: 'street_snap',
    mediaType: 'cinematic',
    prompt: '(best quality, masterpiece, 8k, photorealistic:1.3), (cyberpunk street photography:1.2), (neon night lighting:1.3), (reflections on wet ground:1.1), (futuristic urban backdrop:1.1), vibrant cool tones, cinematic atmosphere',
    negativePrompt: '(worst quality, low quality:1.4), (daylight), (natural), (anime, 3d), ugly, text',
    description: '都市霓虹，赛博质感，夜间写实街摄',
    thumbnail: 'cyberpunk_street.png',
  },
];

// ============================================================
// 导出
// ============================================================

/** 所有风格预设 */
export const VISUAL_STYLE_PRESETS: readonly StylePreset[] = [
  ...STYLES_REAL,
  ...STYLES_ECOMMERCE,
  ...STYLES_COMMERCIAL,
] as const;

/** 分类信息 */
export const STYLE_CATEGORIES: { id: StyleCategory; name: string; styles: readonly StylePreset[] }[] = [
  { id: 'e_commerce', name: '电商快拍', styles: STYLES_ECOMMERCE },
  { id: 'commercial', name: '商业广告', styles: STYLES_COMMERCIAL },
  { id: 'street_snap', name: '潮流街拍', styles: [
    ...VISUAL_STYLE_PRESETS.filter(s => s.category === 'street_snap')
  ] },
  { id: 'real', name: '写实摄影', styles: STYLES_REAL },
];

/** 根据 ID 获取风格 */
export function getStyleById(styleId: string): StylePreset | undefined {
  return VISUAL_STYLE_PRESETS.find(s => s.id === styleId);
}

/** 获取风格的提示词 */
export function getStylePrompt(styleId: string): string {
  const style = getStyleById(styleId);
  return style?.prompt || VISUAL_STYLE_PRESETS[0].prompt;
}

/** 获取风格的负面提示词 */
export function getStyleNegativePrompt(styleId: string): string {
  const style = getStyleById(styleId);
  return style?.negativePrompt || VISUAL_STYLE_PRESETS[0].negativePrompt;
}

/** 获取风格名称 */
export function getStyleName(styleId: string): string {
  const style = getStyleById(styleId);
  return style?.name || styleId;
}

/** 获取风格缩略图路径 */
export function getStyleThumbnail(styleId: string): string {
  const style = getStyleById(styleId);
  return style?.thumbnail || VISUAL_STYLE_PRESETS[0].thumbnail;
}

/** 
 * 兼容旧版：获取风格 tokens（拆分成数组）
 * @deprecated 建议直接使用 getStylePrompt
 */
export function getStyleTokens(styleId: string): string[] {
  const prompt = getStylePrompt(styleId);
  // 简单拆分主要关键词（去除权重标记）
  return prompt
    .replace(/\([^)]*:[0-9.]+\)/g, (match) => match.replace(/:[0-9.]+\)/, ')'))
    .split(',')
    .map(s => s.trim().replace(/^\(|\)$/g, ''))
    .filter(s => s.length > 0)
    .slice(0, 8);
}

/**
 * 根据分类获取风格列表
 * @param categoryId 分类 ID（支持旧版 'animation'/'realistic' 和新版）
 */
export function getStylesByCategory(categoryId: string): StylePreset[] {
  // 兼容旧版分类名称
  const categoryMap: Record<string, StyleCategory[]> = {
    'animation': [], // 已移除
    'realistic': ['real', 'e_commerce', 'street_snap'],
    '3d': [], // 已移除
    '2d': [], // 已移除
    'real': ['real'],
    'e_commerce': ['e_commerce'],
    'commercial': ['commercial'],
    'street_snap': ['street_snap'],
  };
  
  const targetCategories = categoryMap[categoryId] || [categoryId as StyleCategory];
  return VISUAL_STYLE_PRESETS.filter(s => targetCategories.includes(s.category));
}

/**
 * 获取风格描述
 * @param styleId 风格 ID
 */
export function getStyleDescription(styleId: string): string {
  const style = getStyleById(styleId);
  return style?.description || style?.name || styleId;
}

/**
 * 根据风格 ID 获取媒介类型
 * @returns 匹配的 MediaType，未找到时默认返回 'cinematic'（直通，最安全默认值）
 */
export function getMediaType(styleId: string): MediaType {
  const style = getStyleById(styleId);
  return style?.mediaType ?? 'cinematic';
}

/** 媒介类型中文标签 */
export const MEDIA_TYPE_LABELS: Record<MediaType, string> = {
  'cinematic': '电影摄影',
  'graphic': '图形色彩',
};

/** 风格 ID 类型 */
export type VisualStyleId = typeof VISUAL_STYLE_PRESETS[number]['id'];

/** 默认风格 ID */
export const DEFAULT_STYLE_ID: VisualStyleId = 'amazon_model';
