import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

from collections import Counter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud

for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(
    [
        "drug",
        "medication",
        "medicine",
        "tablet",
        "pill",
        "dose",
        "mg",
        "take",
        "taken",
        "taking",
    ]
)


def load_data() -> pd.DataFrame:
    files = ["data/drugsComTrain_raw.csv", "data/drugsComTest_raw.csv"]
    frames = [pd.read_csv(f) for f in files if os.path.exists(f)]

    if not frames:
        print("Brak danych w folderze 'data/'")
        print("https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]  # type: ignore[assignment]
    df = df.rename(columns={"drugname": "drug_name", "usefulcount": "useful_count"})
    df["review"] = df["review"].fillna("").astype(str)
    df["drug_name"] = df["drug_name"].fillna("").str.lower().str.strip()
    return df


def clean_text(text: str) -> str:
    text = re.sub(r"&#\d+;|&\w+;", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower()


def get_sentiment(text: str) -> float:
    return TextBlob(clean_text(text)).sentiment.polarity  # type: ignore[union-attr]


def get_keywords(texts: list[str], top_n: int = 20) -> list[tuple]:
    all_words = []
    for text in texts:
        tokens = word_tokenize(clean_text(text))
        filtered = [w for w in tokens if w not in STOP_WORDS and len(w) > 3]
        all_words.extend(filtered)
    return Counter(all_words).most_common(top_n)


def label_sentiment(score: float) -> str:
    if score > 0.2:
        return "pozytywna"
    elif score < -0.2:
        return "negatywna"
    return "neutralna"


def analyze_drug(drug_name: str, df: pd.DataFrame) -> None:
    query = drug_name.lower().strip()

    mask_partial = df["drug_name"].str.contains(query, na=False)
    subset: pd.DataFrame = df[mask_partial].copy()  # type: ignore[union-attr]

    if subset.empty:
        print(f"\nNie znaleziono leku '{drug_name}' w bazie.")
        return

    print(f"  Analiza leku: {drug_name.upper()}")
    print(f"Znaleziono {len(subset)} recenzji")

    avg_rating: float = float(subset["rating"].mean())  # type: ignore[union-attr]

    print(f"\n OCENY PACJENTÓW:")
    print(f"  Średnia ocena:  {avg_rating:.2f} / 10")
    print(f"  Mediana:        {subset['rating'].median():.1f} / 10")  # type: ignore[union-attr]

    # Sentyment
    print(f"\n ANALIZA SENTYMENTU")
    subset["sentiment"] = subset["review"].apply(get_sentiment)  # type: ignore[union-attr]
    avg_sent: float = float(subset["sentiment"].mean())  # type: ignore[union-attr]
    sent_counts: pd.Series = subset["sentiment"].apply(label_sentiment).value_counts()  # type: ignore[union-attr]

    print(f"  Średni sentyment: {avg_sent:+.3f}  → {label_sentiment(avg_sent)}")
    for label, count in sent_counts.items():
        pct = 100 * count / len(subset)
        bar = "█" * int(pct / 5)
        print(f"  {label:12s}: {count:5d} ({pct:5.1f}%) {bar}")

    # na co chorowali
    print(f"\n NAJCZĘSTSZE WSKAZANIA:")
    conditions = subset["condition"].dropna()  # type: ignore[union-attr]
    conditions = conditions[~conditions.str.contains("</span>", regex=False)]
    conditions = conditions.value_counts().head(8)  # type: ignore[union-attr]
    for cond, cnt in conditions.items():
        pct = 100 * cnt / len(subset)
        print(f"  {str(cond):35s} {cnt:5d} ({pct:.1f}%)")

    # keywordy
    print(f"\n TOP 15 SŁÓW KLUCZOWYCH Z RECENZJI:")
    keywords = get_keywords(subset["review"].tolist(), top_n=15)
    for i, (word, freq) in enumerate(keywords, 1):
        bar = "▪" * min(freq // 10 + 1, 20)
        print(f"  {i:2d}. {word:20s} {freq:5d}x  {bar}")

    # Wykresy
    _plot_analysis(drug_name, subset, keywords, avg_rating, sent_counts)


def _plot_analysis(
    drug_name: str,
    subset: pd.DataFrame,
    keywords: list,
    avg_rating: float,
    sent_counts: pd.Series,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Text Mining Analysis: {drug_name.upper()}", fontsize=16, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Rozkład ocen
    ax1 = fig.add_subplot(gs[0, 0])
    rating_counts = subset["rating"].value_counts().sort_index()
    colors = [
        "#d32f2f" if r <= 3 else "#f57c00" if r <= 6 else "#388e3c"
        for r in rating_counts.index
    ]
    ax1.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="white")  # type: ignore[arg-type]
    ax1.set_title("Rozkład ocen pacjentów")
    ax1.set_xlabel("Ocena (1-10)")
    ax1.set_ylabel("Liczba recenzji")
    ax1.axvline(
        avg_rating, color="navy", linestyle="--", label=f"Średnia: {avg_rating:.1f}"
    )
    ax1.legend()

    # 2. Sentyment
    ax2 = fig.add_subplot(gs[0, 1])
    colors_sent = {
        "pozytywna": "#43a047",
        "neutralna": "#fb8c00",
        "negatywna": "#e53935",
    }
    labels: list[str] = [str(x) for x in sent_counts.index.tolist()]
    values = sent_counts.values.tolist()
    ax2.pie(
        values,
        labels=labels,
        colors=[colors_sent.get(l, "gray") for l in labels],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Podział sentymentu recenzji")

    # 3. Dystrybucja sentymentu
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(subset["sentiment"], bins=30, color="#5c6bc0", edgecolor="white")
    ax3.axvline(0, color="gray", linestyle="--")
    ax3.set_title("Rozkład sentymentu")
    ax3.set_xlabel("Polarność (-1 neg / +1 poz)")
    ax3.set_ylabel("Liczba recenzji")

    # 4. Top słowa kluczowe
    ax4 = fig.add_subplot(gs[1, 0:2])
    words_kw = [w for w, _ in keywords[:12]]
    freqs_kw = [f for _, f in keywords[:12]]
    bar_colors = plt.get_cmap("RdYlGn")(np.linspace(0.2, 0.9, len(words_kw)))
    ax4.barh(words_kw[::-1], freqs_kw[::-1], color=bar_colors)
    ax4.set_title("Top słowa kluczowe w recenzjach")
    ax4.set_xlabel("Częstość")

    # 5. Chmura słów
    ax5 = fig.add_subplot(gs[1, 2])
    all_text = " ".join(subset["review"].tolist())
    all_text_clean = clean_text(all_text)
    filtered_words = [
        w for w in all_text_clean.split() if w not in STOP_WORDS and len(w) > 3
    ]

    if filtered_words:
        wc = WordCloud(
            width=400,
            height=300,
            background_color="white",
            max_words=60,
            colormap="RdYlGn",
        ).generate(" ".join(filtered_words))
        ax5.imshow(wc, interpolation="bilinear")
        ax5.axis("off")
        ax5.set_title("Chmura słów")

    output_file = f"analysis_{drug_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n Wykres zapisany jako: {output_file}")
    plt.show()


if __name__ == "__main__":
    print("Ładowanie danych...")
    df = load_data()

    while True:
        drug = input("Lek: ").strip()
        if drug:
            analyze_drug(drug, df)
