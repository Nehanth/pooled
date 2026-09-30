import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		// `eyebrow`: the small section label above the H1 (the sidebar group name).
		schema: docsSchema({ extend: z.object({ eyebrow: z.string().optional() }) }),
	}),
};
