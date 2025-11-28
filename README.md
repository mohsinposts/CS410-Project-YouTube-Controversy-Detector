# YouTube Controversy Detector

This project analyzes YouTube comments to measure how audiences react to different videos and identify which ones are truly controversial. Rather than simply measuring whether a video is positive or negative overall, the system detects when viewers are genuinely split in their opinions.

**Project context:** CS410 Fall 2025 course project on information retrieval and text mining. Originally planned to use the YouTube Data API, but pivoted to scraping-based approaches (yt-dlp and youtube-comment-downloader) so graders can run the code in Google Colab without needing API keys or authentication.

**Main file:** YouTube_Controversy.ipynb

---

## What the tool does

For a given list of YouTube channels, the notebook:

1. Uses yt_dlp to fetch the most recent videos from each channel.
2. Uses youtube-comment-downloader to scrape the top comments for each video.
3. Runs each comment through a pretrained sentiment model from Hugging Face, cardiffnlp/twitter-roberta-base-sentiment-latest.
4. Labels each comment as negative, neutral, or positive.
5. Aggregates sentiment counts per video.
6. Computes a controversy score per video based on the balance of positive and negative comments.
7. Prints a summary for each video and generates plots that show:
   - The most controversial videos
   - Video level controversy per channel
   - Average controversy per channel

This makes it easy to see which videos have one sided support and which ones attract real disagreement.

---

## How controversy is measured

The core insight: A video is controversial when viewers are genuinely divided, not just when some people dislike it. A video with 90% positive and 10% negative comments isn't controversial—it's just popular with a few critics. True controversy requires substantial disagreement on both sides.

### The controversy formula

For each video, we count:
- **P** = positive comments  
- **N** = negative comments  
- **T** = P + N (non-neutral comments)

Neutral comments are tracked for completeness but don't factor into controversy scoring since they represent fence-sitting rather than strong opinions.

**Step 1: Check for meaningful opposition**

Before calculating controversy, we verify both sides have real presence:
- Calculate p_ratio = P / T and n_ratio = N / T
- Require both ratios ≥ 0.25 (25%)

If either side falls below 25%, controversy = 0. This prevents labeling videos as controversial when they're actually just one-sided with a vocal minority.

**Example:** PewDiePie video with 50 positive, 8 negative → 8/58 = 14% negative → Not controversial, just popular.

**Step 2: Calculate balance score**

For videos passing the threshold:

controversy = 4 × P × N / T²

This formula peaks at 1.0 when opinions are perfectly split (50/50) and approaches 0 as one side dominates.

**Example:** Video with 35 positive, 32 negative → Both sides ≥25% → controversy ≈ 0.99 → Highly controversial.

**Step 3: Assign verdict**

Each video gets labeled:
- **"Controversial/Mixed Opinions"** if controversy > 0
- **"Not Controversial/One-Sided Opinions"** if controversy = 0

This binary classification makes results easy to interpret at a glance.

---

## Implementation details

### Architecture overview

The system is built as a Python notebook (YouTube_Controversy.ipynb) with a modular pipeline:

1. **Data collection** → 2. **Sentiment analysis** → 3. **Controversy computation** → 4. **Visualization**

### Key implementation decisions

**Why these libraries?**
- **yt-dlp** instead of YouTube Data API: No authentication required, making it easier for graders to run. Uses the `extract_flat` option to quickly grab video metadata without downloading actual video files.
- **youtube-comment-downloader**: Scrapes comments without API keys. Configured to fetch top comments first using `SORT_BY_POPULAR` since highly-voted comments better represent overall audience reaction.
- **cardiffnlp/twitter-roberta-base-sentiment-latest**: Initially tried VADER, but it struggled with YouTube-style language (slang, emojis, sarcasm). This Twitter-trained RoBERTa model handles informal text much better, though it's slower due to transformer architecture.

### Module breakdown

**Data collection functions:**
- `get_recent_videos_from_channel()`: Uses yt-dlp with `player_client=["android"]` to avoid bot detection. Extracts video IDs from channel URLs and returns the N most recent videos.
- `download_comments()`: Scrapes top comments using youtube-comment-downloader with a configurable limit per video.
- `get_video_title()`: Fetches video metadata with error handling for age-restricted or deleted videos.

**Sentiment analysis:**
- `analyze_sentiment()`: Tokenizes comments (max 512 tokens), runs through the RoBERTa model, applies softmax to get probability distributions, and maps to categorical labels plus numeric scores (-1, 0, +1).
- `run_sentiment_pipeline()`: Applies sentiment analysis across all comments in the dataframe using pandas `.apply()`.

**Controversy metrics:**
- `compute_controversy()`: Groups comments by video and sentiment label, then applies the controversy formula. Only labels videos as controversial if both positive and negative sentiments each represent at least 25% of non-neutral comments.
- `categorize_video()`: Simple threshold-based categorization into "Controversial/Mixed Opinions" vs "Not Controversial/One-Sided Opinions".

**Visualization:**
- Three matplotlib/seaborn plots showing different views of the data:
  - Stacked bar chart of top controversial videos with sentiment breakdowns
  - Scatter plot showing controversy distribution per channel
  - Bar chart ranking channels by average controversy

### Data flow

Raw YouTube data → Comment text → Sentiment labels → Aggregated counts → Controversy scores → Visual reports

---

## How to run the notebook

You can run everything in Google Colab. No API keys are required.

1. Open the notebook on GitHub  
   - Go to this repository in your browser.  
   - Click on YouTube_Controversy.ipynb.

2. Open in Google Colab  
   - At the top of the notebook page on GitHub, click the "Open in Colab" button in the top left area.

3. Run all cells  
   - In Colab, go to Runtime → Run all, or use the run all button.  
   - Wait until all cells finish. The notebook will print a section named VIDEO CONTROVERSY ANALYSIS and show the plots at the bottom.

---

## Customizing the analysis

At the bottom of the notebook you will see something like:

```python
channels = [
    "https://www.youtube.com/@ChannelOne",
    "https://www.youtube.com/@ChannelTwo",
]

data, video_titles, channel_names = process_channels(
    channels,
    videos_per_channel=3,
    comment_limit=50
)

df, stats = run_full_analysis(data, video_titles, channel_names)
```

You can change what is analyzed by editing this block.

### Change the channels

Edit the channels list with any public YouTube channel handles you want, for example:

```python
channels = [
    "https://www.youtube.com/@PewDiePie",
    "https://www.youtube.com/@SHNEAKO",
    "https://www.youtube.com/@SomeOtherCreator",
]
```

Then rerun the cells that define channels, the process_channels call, and run_full_analysis.

### Change number of videos per channel

Adjust the videos_per_channel argument:

```python
videos_per_channel=1    # only the latest video from each channel
videos_per_channel=3    # default
videos_per_channel=5    # more videos per channel
```

Higher values give more data and longer runtime.

### Change number of comments per video

Adjust comment_limit:

```python
comment_limit=20        # fewer comments, faster, but noisier
comment_limit=50        # default
comment_limit=100       # more comments, slower, more stable
```

Rerun the processing and analysis cells after changing this.

---

## Troubleshooting

### yt_dlp "Sign in to confirm you are not a bot" error

Sometimes YouTube may return an error like:

"Sign in to confirm you are not a bot. Use --cookies-from-browser or --cookies for authentication."

If this happens:

1. Open GitHub and the Colab notebook in a private or incognito window, then try running again.  
2. The code already catches some DownloadError cases when fetching video titles. If a video cannot be accessed, it will be skipped and the rest of the analysis will still run.

If a specific channel or video keeps causing problems, you can remove that channel from the channels list.

---

## Tutorial video

A short usage tutorial for this project is available on Illinois Media Space.

Tutorial video link: https://mediaspace.illinois.edu/media/t/1_elotad73
