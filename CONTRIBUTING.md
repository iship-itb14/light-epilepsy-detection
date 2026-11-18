# Editing the website: a guide for the team

This repo is our Neural Networks final project website, **Listening for Seizures**. It is plain HTML, CSS and JavaScript hosted on GitHub Pages. There is no framework and no build step: edit a file, refresh the browser, and you see the change.

Please read the whole guide once before your first edit. The design rules at the bottom are what keep the site from looking generic, so follow them closely.

---

## 1. Get the site running on your computer

1. Accept the collaborator invite from GitHub (check your email).
2. Clone the repo:
   ```
   git clone https://github.com/iship-itb14/light-epilepsy-detection.git
   cd light-epilepsy-detection
   ```
3. Start a local server. Opening `index.html` by double-clicking will **not** work, because the globe loads its map data from a file.
   ```
   python3 -m http.server 8000
   ```
4. Open http://localhost:8000 in your browser.

## 2. How the files are organised

```
index.html                  Introduction (home page)
data.html                   Data & preparation      (sections laid out, content to be written)
models.html                 Models / methods        (sections laid out, content to be written)
results.html                Results                 (sections laid out, content to be written)
conclusions.html            Conclusions             (opening film in place, text to be written)
about.html                  About us                (photos and bios for each of us)

assets/css/site.css         ONE shared stylesheet for every page
assets/js/intro.js          Globe, scroll story and animations for the home page
assets/js/media.js          Plays looping videos only while on screen, with a Pause button
assets/js/footer.js         The moving brainwave art in every page's footer
assets/data/land-dots.json  Map points for the globe (do not edit by hand)
assets/video/               Videos (mp4)
assets/img/                 Images (jpg / webp / png)

data/raw/                   Where the raw recordings come from and how to download them
data/clean/                 Cleaned data, if small enough for GitHub
code/                       Notebooks and scripts for cleaning and models
results/                    Saved figures and metrics shown on the Results page
```

The professor requires links to all raw data, cleaned data and code. Put them in `data/` and `code/` and link to them from the matching page.

## 3. The everyday workflow

Work on a branch so two people never overwrite each other.

```
git pull                         # always start from the latest version
git checkout -b results-page     # name the branch after what you're doing
# ...edit files, check them at localhost:8000...
git add .
git commit -m "Add confusion matrix to results page"
git push -u origin results-page
```

Then open a **Pull Request** on GitHub and ask one teammate to look at it before merging. After it merges, GitHub Pages republishes the live site within about a minute.

**Who owns what.** Team: Sejal Hukare, Ishita Pawar, Swejal Gabhane. Agree in the group chat who owns each page before you start editing it, so two people never change the same file at once. Everyone edits their own entry on the About us page.

`assets/css/site.css` is shared. If you need to change it, say so in the group chat first.

## 4. Filling in your page

Data & prep, Models and Results already have their sections laid out to match the rubric. Each section has:
- a grey note (`<p class="todo">`) saying what belongs there. Replace it with your writing and change the tag to a plain `<p>`.
- a dashed box (`<div class="slot">`) where a figure goes. Replace it with your image:
  ```html
  <figure>
    <img src="assets/img/results-confusion.png" alt="Confusion matrix for the lightweight model" width="1600" height="900">
    <figcaption>What the figure shows, in one sentence.</figcaption>
  </figure>
  ```

Your About us entry is in `about.html`. Replace "Photo to come", your bio and your project role.
- **Photo:** a portrait with the background removed (a transparent PNG) looks best and matches Sejal's. Send it to Sejal, or save it as `assets/img/team-<firstname>.webp` and swap it into your `<figure class="portrait">`.

## 5. Making a new page

1. Copy `index.html` and rename it, for example `results.html`.
2. Keep the `<nav>` and `<footer>`. Delete everything inside `<main>`.
3. In the nav, move `aria-current="page"` to your page's tab. That attribute draws the small ember line under the current tab.
4. Remove the `<script src="assets/js/intro.js">` line, which only runs the home page's globe. Keep `media.js` if your page has a video.
5. Build your content inside `<main>` using the existing classes: `.sec` for a section, `.wrap` for the 12-column grid, `.prose` for body text, `h2` for section titles.
6. Add your page to the nav on **every** page, and keep the `<script src="assets/js/footer.js">` line so the footer art draws.

## 6. Adding images and video

- **Video:** use mp4 (H.264), no sound, under about 8 MB. Compress with:
  ```
  ffmpeg -i input.mov -vf "scale=1920:-2" -c:v libx264 -preset veryfast -crf 26 -an -movflags +faststart assets/video/name.mp4
  ```
  GitHub rejects any file over 100 MB.
- **Looping silent video** (like the two animations already on the site): add `data-loop` to the `<video>` and put `<button class="vctl" type="button">Pause</button>` in the same frame. `media.js` then plays it only while it's on screen, never for visitors who turn off motion, and lets anyone pause it. Copy the markup from `index.html`.
- **Images:** export at most 2000 px wide, as jpg or webp, under about 500 KB.
- **Only use media we are allowed to use:** our own, or free-licence sources such as Pexels or Unsplash. Write the source in the figure caption.
- **Every image needs alt text** describing what it shows.
- **Footage of a real seizure must be handled with care.** Put a short content note before it, never autoplay it, and let the viewer choose to play it.

## 7. Design rules (please don't break these)

**Colour.** Every colour has one job. Use the variables in `site.css`, never new hex codes.

| Variable | Colour | Job |
|---|---|---|
| `--forest` | #0F1A15 | The page background (about 90% of the screen) |
| `--bone` | #E7E3D6 | All main text |
| `--moss` | #8E9B8A | Captions, sources, secondary text |
| `--canopy` | #17261F | Raised surfaces: video frames, the globe |
| `--ember` | #E3A33B | **Only** seizures, the live signal, and "you are here." If you are tempted to use it for decoration, don't. |

**Type.**
- Headings use **Newsreader** (light weight). It is the only display font.
- Body text uses **Instrument Sans**, 16–18 px. The site uses 17 px.
- Use sentence case everywhere. No ALL-CAPS labels, and no single highlighted word inside a headline.
- Keep lines under about 80 characters. `.prose` already caps this.

**Layout.**
- All spacing comes from the 8-point scale (`--s1` = 8px, `--s2` = 16px, `--s3` = 24px, and so on). Never type a pixel value that isn't a multiple of 8.
- Don't stack everything down the middle. Place content off-centre on the 12-column grid, with captions or notes in a side column. The existing sections show how.

**Never:**
- gradients
- drop shadows
- rounded "cards" for every block
- emoji
- stock icons
- a fade-up animation on every section
- an arrow ("→") on links

**Accessibility.** The site already respects "reduce motion", has visible keyboard focus, and works on phones. After every change, check your page at phone width in your browser's dev tools.

## 8. Writing rules (from the professor's rubric)

- **Introduction and Conclusions:** 100% non-technical, at least 5 paragraphs, each paragraph 7–10 full sentences, with images. Write so that any reader can follow it. No model names, no metrics.
- **Data & prep:** where the data came from, what it looks like, every cleaning and labelling step, and plots. Link to the raw data, the cleaned data and the cleaning code.
- **Models:** what each network is, why we chose it, how it works, and an architecture diagram for each.
- **Results:** link to the model code, accuracy and other metrics, **at least one confusion matrix**, and what the numbers mean.
- **Tone:** professional and academic. Cite sources in the footer's numbered list.
- **Originality:** the project must be our own work. Write in your own words and cite anything you borrow (the penalty is at least 50%).

## 9. Before you open a pull request

- [ ] The page works at localhost:8000 and at phone width
- [ ] No new colours, fonts or gradients
- [ ] Images have alt text, and media sources are credited
- [ ] Links to data and code work
- [ ] Nav links are updated on every page
