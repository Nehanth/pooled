# Pooled design language

How the room page (`p2p.html` + `room.js`) and the site look, written down so changes match. Read this before touching UI. Sources of truth: the room's `:root` and `<style>` in `p2p.html`, `site/css/tokens.css`, `site/css/site.css`, and [final-polish.md](final-polish.md).

## 0. Facts first

1. **There is no dark theme.** `:root` sets `color-scheme: light` and nothing reads `prefers-color-scheme` or a `data-theme` attribute. The only dark surfaces are deliberate and fixed: the Code pill `#0B0D14`, the "this device" button (`--ink`), toasts and tooltips (`--text` background, white text), and the compute screen (`#000`, `--game`, `--on-game`). Don't add a dark theme to one panel; a dark theme would have to cover the whole page.
2. **Header alignment.** In the room bar (>=821px) everything in the row is 34px tall and shares one vertical centre. `#mode-bar` (Chat | Code) is 28px buttons + 3px padding. Its neighbours (`#api-open`) take the same outer height in every place the switch sits: 34 in the header, 38 in the phone strip, 42 on coarse-pointer phones.
3. **Examples to copy from:** Invite (`.sheet.invite`, a centred sheet with one hero object), Room settings (`.menu-pop`, a left-aligned sectioned panel), Serve API (`#api-sheet`, an anchored popover with numbered steps that becomes a bottom sheet on phones).

## 1. Tokens (use names, never raw colours)

Canonical names (landing) with room aliases in brackets. All values are in `site/css/tokens.css`, and the room mirrors them in its own `:root`.

| Role | Token | Value |
|---|---|---|
| Page background | `--bg` | #F6F5F1 |
| Raised/quiet surface, header bg in room, input wells | `--bg-2` (`--panel-3`) | #FBFAF7 |
| Well / track / segmented background | `--bg-3` (`--panel-2`) | #EFEDE7 |
| Card, sheet, pill, chip | `--white` (`--panel`) | #FFFFFF |
| Text | `--ink` (`--text`) | #14161D |
| Secondary text, icons | `--ink-2` | #4B4E58 |
| Muted text, labels | `--ink-3` (`--muted`) | #5E616B |
| Faintest text / asks-only device | `--ink-4` | #6B6E78 |
| Default 1px line | `--rule` (`--border`) | #E4E2DA |
| Soft inner line, dividers, well outline | `--rule-2` | #ECEAE3 |
| Strong line, secondary-button outline, hover borders | `--rule-3` (`--border-2`) | #D6D3C9 |
| Placeholder dot, scrollbar | `--faint` | #C4C0B1 |
| The one accent | `--blue` (`--accent`) | #2A45E0 |
| Accent text, hover of primary | `--blue-text` / `--blue-600` (`--accent-text`, `--accent-hover`) | #2239C8 |
| Accent tint (selected card bg, focus halo) | `--blue-soft` / `--blue-50` (`--accent-soft`) | #EAEDFE |
| Ramp | `--blue-200` (`--a3`) / `--blue-400` (`--a2`) / `--blue-700` | #B9C6FF / #7C8FFF / #1C33B8 |
| OK / error | `--ok` / `--del` (`--err`) | #1A7F37 / #B42318 |
| Attention | `--warn` = the accent (never yellow) | |

Type: `--display` Funnel Display (titles only, weight 400), `--sans` Geist (400/500/600), `--mono` Geist Mono (400/500).
Text sizes: `--t-xs 11`, `--t-sm 12`, `--t-md 13`, `--t-base 14`, `--t-lg 15`, `--t-xl 17`. Display: `--d-sm 22`, `--d-md 26`, `--d-lg 36`. No half-pixel sizes, and mono only at 11 or 12 (the one exception is big codes such as the 26px invite code).
Radius: `--r-xs 4` (inline code, status chips), `--r-sm 8` (buttons, inner pills, icon buttons), `--r-md 12` (wells, rows, linkrows, popovers, toasts), `--r-lg 16` (sheets, panels, menu-pop), `--r-pill 999` (switch, chips).
Motion: `--ease: cubic-bezier(.2,.7,.2,1)` for everything. `--glide: cubic-bezier(.65,0,.25,1)` for the Chat|Code pill slide (0.55s). The room hardcodes the glide curve, so use `var(--glide)` if you touch it (the room doesn't define `--glide` yet; add it to the room `:root` or keep the literal).

Allowed raw values (these already exist and have no token): shadow rgba built on ink `rgba(20,22,29,a)`, the accent glow `rgba(42,69,224,.7)` / `--blue-glow`, `#fff` for text on blue or black, the Code pill `#0B0D14` with its glow `rgba(110,134,255,.6)`, and the header glass `rgba(246,245,241,.85)`. Anything new must be a token or `color-mix()` of tokens.

## 2. Spacing, borders, shadows, motion

- **Spacing** is a loose 2/4 grid: gaps are 2 (inside a well), 6, 8, 10, 12, 16, 18, 20, 24. Sheet padding is `28px 24px 22px` (desktop) and `30px 20px 20px` (phone sheet). Menu sections use `16px 18px 18px`. Section-to-section in a sheet is 16-20px, label-to-control is 8px.
- **Lines**: always 1px. Use `border` for outer edges (sheet: `--border`) and `box-shadow: inset 0 0 0 1px var(--rule-2)` for wells and `.hbtn`, so height never changes with the border. Dividers inside panels are `border-top: 1px solid var(--rule-2)`.
- **Shadow ladder** (all ink-tinted, negative spread, soft):
  - inner pill: `0 1px 2px rgba(20,22,29,.08), 0 0 0 1px rgba(20,22,29,.05)`
  - popover / menu: `0 1px 2px rgba(20,22,29,.05), 0 18px 40px -16px rgba(20,22,29,.28)` (menu: `0 24px 60px -20px … .32`)
  - sheet: `0 24px 60px -20px rgba(20,21,26,.35)`
  - primary blue: `0 1px 0 rgba(255,255,255,.18) inset, 0 6px 16px -8px rgba(42,69,224,.7)`
  - focus/selected halo: `0 0 0 3px var(--accent-soft)` with `border-color: var(--accent)`
- **Motion**:
  - button colour/border/shadow `.15s var(--ease)`, press `transform: scale(.98)` over `.12s`
  - pill slide `.55s` glide, label colour `.35s`
  - popovers `popin .22s` (`translateY(-4px)`, fade)
  - desktop sheet `rise .3s` (`translateY(8px)`, fade), phone sheet `sheetup .32s` (from `translateY(100%)`)
  - switch knob `.25s`, toasts in `.45s`
  - Always honour `prefers-reduced-motion` (a block exists at ~line 1385; add new animations there).

## 3. Component recipes

### 3.1 Segmented switch with a sliding pill (`#mode-bar`, the "Chat | Code" look)
```
well:   position:relative; display:flex; padding:3px; border-radius:12px;
        background:var(--panel-2); box-shadow:inset 0 0 0 1px var(--rule-2);
pill:   ::before; position:absolute; top:3px; left:3px; width:W; height:H; border-radius:8px;
        background:var(--panel); box-shadow:0 1px 2px rgba(20,22,29,.08),0 0 0 1px rgba(20,22,29,.04);
        transition:transform .55s var(--glide), background-color .3s ease, box-shadow .3s ease;
option: width:W; height:H; border:0; background:transparent; font:500 14px/1 var(--sans); color:var(--muted);
        selected -> color:var(--text); hover -> color:var(--text) (no background); :active no scale.
Code selected: pill #0B0D14, glow 0 0 16px -3px rgba(110,134,255,.6), word #fff, sparkles (data-twinkle).
```
Sizes: phone strip W=84 H=32 (outer 38). **Header W=68 (`--mw`) H=28 (outer 34), 13px.** Coarse phones H=36 (outer 42). The outer radius (12) = inner radius (8) + padding (3) + 1. Keep that relationship.

### 3.2 Header controls (room bar, 56px tall, bg `--panel-3`, bottom `1px --rule-2`)
- Order: logo, `|` rule, Room CODE (`#room-badge`: 13px mono .14em tracking, "Room" in sans muted, left border `--rule-2`, height 32), Chat|Code (34), device chips (`.pchip` 30px, radius 8, `--border`, self chip border `--blue-200`), spacer, `#hdr-sum` (12px mono), Invite (primary, 34px), this device (34px black square), settings (34px), GitHub (36px).
- `.hbtn` secondary: `height:34px; padding:0 12px; radius 8; font 500 13px; bg --panel; box-shadow inset 0 0 0 1px var(--border-2)`, hover `bg --panel-3` with `inset … var(--muted)`.
- Icon buttons: 34x34, radius 8, transparent, colour `--ink-2`, hover `--panel-2`. They get a tooltip through `data-tip` (dark `--text` bubble, 12px, radius 8).
- **Rule: every control in one header row has height 34 (desktop) / 36 (<=820px). Anything next to `#mode-bar` in the header matches its 34px outer height and its vertical centre.** Gap between header groups is 12 (the header's `gap`). The mode-bar has `margin-left:4px`; a neighbour uses 8px max, or the header gap alone.

### 3.3 Buttons
- Primary (blue is the **only** primary): `bg --accent; color #fff; radius 8; font 500 14px; padding 12px 20px` (compact: `height 34; padding 0 14px; 13px`). Hover `--accent-hover`. In the header it carries the blue glow shadow. Disabled: `--panel-2` bg, `--muted` text, opacity 1.
- Secondary: white, 1px `--rule-3` (landing `.btn-line`) or the `.hbtn` inset ring. Hover `--bg-2` with an `--ink-3` ring.
- Quiet/ghost (toolbars): transparent, `--muted` text, 12px, `padding 6px 10px`, hover `--panel-3`.
- Black (`--text`) only for Stop and destructive-neutral. Destructive text uses `--del`.
- Every button: sentence case, `transform: scale(.98)` on press, `:focus-visible` = `2px solid var(--accent)` outline, offset 2.

### 3.4 Sheets and overlays (`.overlay` > `.sheet`)
- Overlay: fixed, z 70, `rgba(20,22,29,.32)` + `backdrop-filter: blur(4px)`, padding 16, centred.
- Sheet: `bg --panel; 1px --border; radius 16; padding 28/24/22; width min(400px,100%)` (`.wide` 680), shadow as above, `rise .3s`, scrolls inside (`max-height: calc(100dvh - 32px)`).
- Close `.x`: 32x32 at top/right 12, radius 8, `--muted`, hover `--panel-2`, 14px X icon with 1.4 stroke.
- Title `h2`: `400 26px/1.1 var(--display)`, tracking -.025em. Lede `.k`: 14px/1.5 `--ink-2`, max 320px, `text-wrap: balance`.
- **<=640px** it becomes a bottom sheet: full width, radius `16 16 0 0`, no bottom border, 36x4 grab bar (`--border-2`, top 7), `sheetup`, bottom padding `max(20px, env(safe-area-inset-bottom))`, row buttons 44px tall and equal width.
- Invite is the model centred sheet: title, lede, one hero object, a `.linkrow` with copy.
- Room settings (`.menu-pop`) is the model for a **left-aligned, sectioned** panel: sticky head (22px display title plus a 13px muted sub), `.ms` sections split by `--rule-2`, a 13px `--ink-3` section label (h3, weight 500, **not uppercase**), each setting = 14px/500 label + 12px muted help + control. Use the `.ms > h3` style for section labels, never uppercase tracked labels.

### 3.5 Copy row / code (`.linkrow`)
`display:flex; gap:6px; padding:5px 5px 5px 12px; border:1px solid var(--border-2); radius 12; bg --panel-3`. Text is `400 12px/1.2 var(--mono)` in `--ink-2` or `--text`, single line with ellipsis (share) or horizontal scroll (commands, as in Serve API). Button is compact primary, 34px, 13px. Inner radius 8 + padding 5 ≈ outer 12.
Code blocks (chat `pre`): mono 12px, `bg --panel-3`, `1px --rule-2`, radius 12, padding `10px 12px`, `overflow-x:auto` (**scroll, don't wrap** multi-token commands like curl). Inline code radius 4. A prompt glyph or `$` in `--muted` is fine. No syntax colours beyond ink, ink-2 and muted, plus accent for at most one highlight.

### 3.6 Other controls
- `.seg` (settings): same well as 3.1 but with static options, `gap 2`, option 32px high, 13px. Selected = white pill with the inner-pill shadow. `.seg.chips` = 30px pills, radius 999.
- Switch: 40x24 track, radius 999, off `--border-2`, on `--accent`. 18px white knob with `0 1px 2px rgba(20,22,29,.25)`, `.25s` travel. Row: label flex 1, 13-14px `--ink-2`, gap 12-16.
- Tabs in the Code pane (`#code-out-tabs`): the same well, buttons 28px, 13px, `.on` = white pill. Phone bottom tabs are 56px with icon over a 12px label.
- Model-picker cards: `1px --border` rows, radius 12. Selected = `--accent` border + `--accent-soft` bg + blue radio. Size is 12px mono muted at the right.
- Device chip / popover (`.pop`): 272px, padding 14, radius 12, popover shadow, 30x30 icon tile (radius 8, `--panel-2`), mono 11-12px stats.
- Toasts: dark `--text` chip, radius 12, 13px/500 white, a 7px dot in `--sw`/`--blue-400`. Presence toasts are white with a `--rule` ring. The Copy action should toast (`copyText()` already does).
- Menu action rows (`.menu-acts`): 32px icon tile (radius 8, `--panel-2`) + 14px/500 label + 12px muted sub, row radius 12, hover `--panel-3`.
- Icons: inline SVG, `stroke: currentColor`, stroke 1.4 at 16px+ and 1.3 at 12-14px, round caps and joins, `aria-hidden`.

## 4. Do / don't

Do:
- Match neighbours exactly: same height, same vertical centre, same radius family (12 outer / 8 inner). In the header, 34px.
- A control next to a segmented switch is either (a) a quiet secondary control (`.hbtn`-like, same outer height and radius as the switch, apart by the header gap), or (b) part of the switch family with the full pill logic. Never a pill-less well that looks like a dead segment (see `#api-open`).
- Put "the one action" first and make it big (a command in a linkrow with the primary Copy). Then reference info as copyable mono rows (each with a quiet copy icon), examples in a scrolling code block, host-only settings in a footer on `--panel-3` apart from the steps, then a 12px muted note. The Serve API popover (`#api-sheet`) is the worked example.
- Left-align dense content (endpoints, code). Centre only a short title and lede, and pick one per sheet.
- Use `--display` only for the sheet title (22 in a panel head, 26 in a sheet).
- Use mono for codes, URLs, ports, commands and numbers only. Labels stay in Geist.
- Use sentence case, no em dashes, lowercase device names, and model names like `Qwen3.6 35B MoE`.
- Dialogs and popovers: `role="dialog"` (`aria-modal="true"` when it blocks), `aria-labelledby`, focus to the first useful control (Copy) or close on open, trap Tab, and **return focus to the opening button on Esc, on scrim click and on close**.
- Check 390px: bottom sheet, 44px touch targets, no horizontal page scroll (long code scrolls inside its block, with a fade at the edge), controls in `.mode-row` at equal height (38, or 42 on coarse pointers).

Don't:
- Don't use raw hex for new things, yellow or amber for attention, or a second accent colour. Don't use gradients except the existing glow and fades.
- Don't use shadows heavier than the sheet ladder, drop shadows on flat wells, or 2px borders.
- Don't make a panel look like a third room mode next to Chat | Code (Serve API opens a popover, per `serve.md`).
- Don't use uppercase tracked labels (the room uses 13px `--ink-3` section heads).
- Don't wrap commands mid-token (`completion` / `s`): code scrolls sideways.
- Don't add a dark mode to the panel alone.
- Don't use half-pixel font sizes, or mono at 13-14px in body text.
- Don't restyle global `button` rules. Scope new rules to your component, and remember the global `button` hover (`--panel-3` bg and `--border-2` border) and press scale apply unless you override them.

## 5. What makes it feel "Pooled"

- Warm off-white paper (`--bg`) with pure white cards floating on it. Hairline rules instead of boxes. Soft ink-tinted shadows that fall long and low (`-16/-20px` spread).
- One saturated blue, used sparingly, for the primary action, selection and "live". Everything else is warm greys.
- Three voices: Funnel Display for calm large titles, Geist for UI, Geist Mono for the machine bits (room code, GB, URLs), often small and muted.
- Wells with a white pill: selection is a physical object that slides (the glide curve). Code mode turns the pill black with a blue glow and twinkles.
- The dots logo and "working" dots (`.cdots`, blue ramp) are the motion motif. Animations are short and eased, and nothing bounces.
- Copy is plain, short, sentence case, and talks about "this room", "your computer", "devices".

## 6. Checklist for a UI change

1. Header at 1280: controls next to `#mode-bar` (e.g. `#api-open`) have the same `getBoundingClientRect()` `top` and `height` (34).
2. Phone at 390: the same in `.mode-row` (38, or 42 on coarse pointers). Sheets are bottom sheets and nothing overflows (`document.documentElement.scrollWidth === 390`).
3. Keyboard: Tab reaches the button, Enter opens, focus is inside, Tab stays inside, Esc closes, focus is back on the button.
4. Tokens only (grep the diff for `#[0-9a-f]{3,6}` outside the allowed list).
5. Host and guest views both look right, empty and full (e.g. no API clients and one or more).
6. Reduced motion is honoured. `deno test --allow-read --no-check tests/unit` and `npm run check` pass.
