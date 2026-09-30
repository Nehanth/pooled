import { defineEcConfig } from '@astrojs/starlight/expressive-code';

// Syntax colours limited to the Pooled palette (language.md 3.5): ink, ink-2, muted, and the accent
// for one role. Greys in the theme (comments, punctuation) stay; min-light's purple (functions,
// commands) becomes the accent and every other hue becomes ink-2.
const PALETTE = { light: { accent: '#2239C8', other: '#4B4E58' }, dark: { accent: '#9AABFF', other: '#C3C8D6' } };
const ACCENT_SRC = ['#6f42c1', '#b392f0'];
function saturation(hex) {
	const n = parseInt(hex.slice(1, 7), 16), r = (n >> 16) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
	const max = Math.max(r, g, b), min = Math.min(r, g, b);
	return max === 0 ? 0 : (max - min) / max;
}

export default defineEcConfig({
				themes: ['min-light', 'min-dark'],
				useStarlightDarkModeSwitch: true,
				// Shell and output blocks get the plain frame: no empty terminal title bar.
				defaultProps: { frame: 'code' },
				customizeTheme(theme) {
		const pal = theme.type === 'dark' ? PALETTE.dark : PALETTE.light;
					for (const rule of theme.settings) {
						const fg = rule.settings?.foreground?.toLowerCase();
						if (!fg || !/^#[0-9a-f]{6}/.test(fg)) continue;
						if (ACCENT_SRC.includes(fg.slice(0, 7))) rule.settings.foreground = pal.accent;
						else if (saturation(fg) > 0.25) rule.settings.foreground = pal.other;
					}
					return theme;
				},
				styleOverrides: {
					borderRadius: '12px',
					borderColor: 'var(--p-rule)',
					borderWidth: '1px',
					codeFontFamily: 'var(--__sl-font-mono)',
					codeFontSize: '13px',
					codeLineHeight: '1.7',
					codePaddingBlock: '14px',
					codePaddingInline: '16px',
					codeBackground: 'var(--p-code-bg)',
					uiFontFamily: 'var(--__sl-font)',
					frames: {
						shadowColor: 'transparent',
						editorTabBarBackground: 'var(--p-code-bg)',
						editorActiveTabBackground: 'var(--p-code-bg)',
						editorActiveTabIndicatorTopColor: 'transparent',
						editorActiveTabIndicatorBottomColor: 'var(--p-ink)',
						editorTabBarBorderBottomColor: 'var(--p-rule)',
						terminalTitlebarBackground: 'var(--p-code-bg)',
						terminalBackground: 'var(--p-code-bg)',
						terminalTitlebarBorderBottomColor: 'var(--p-rule)',
						terminalTitlebarDotsOpacity: '0',
						// copy button: a solid square in the code's own colour, so text never shows through it
						inlineButtonBackground: 'var(--p-code-bg)',
						inlineButtonBackgroundIdleOpacity: '1',
						inlineButtonBackgroundHoverOrFocusOpacity: '1',
						inlineButtonBackgroundActiveOpacity: '1',
						inlineButtonForeground: 'var(--p-ink-3)',
						inlineButtonBorder: 'var(--p-rule)',
						inlineButtonBorderOpacity: '1',
						tooltipSuccessBackground: 'var(--p-ink)',
					},
				},
			});
