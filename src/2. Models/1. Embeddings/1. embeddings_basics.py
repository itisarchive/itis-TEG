"""
🚀 Text Embeddings - Interactive Educational Journey
====================================================

Welcome to your hands-on exploration of text embeddings! This script takes you
through the fundamental concepts of how words become vectors, using practical
examples that you can run, modify, and experiment with.

🎯 What You'll Learn:
- What embeddings are and how they work (dimensions, magnitude, statistics)
- Similarity metrics and their interpretation (cosine, euclidean, dot product)
- Counterintuitive patterns in language relationships (the cat-dog mystery)
- How context affects meaning in vector space
- Semantic clustering and 2D visualization with PCA
- Similarity heatmaps for comprehensive relationship matrices
- The famous king-queen analogy and vector arithmetic

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with openai, numpy, matplotlib, seaborn, scikit-learn
"""

import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class WordPair:
    first: str
    second: str


@dataclass(frozen=True)
class AnalogyTestCase:
    source_word: str
    subtract_word: str
    add_word: str
    expected_target: str


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def fetch_embedding(
        text: str,
        *,
        client: AzureOpenAI,
        cache: dict[str, list[float]],
        model_name: str = "text-embedding-3-small",
) -> list[float]:
    """Generates an embedding vector for the given text using Azure OpenAI's API, with caching."""
    if text in cache:
        return cache[text]

    cleaned_text = text.replace("\n", " ")
    embedding_response = client.embeddings.create(input=[cleaned_text], model=model_name)
    embedding_vector = embedding_response.data[0].embedding
    cache[text] = embedding_vector
    print(f"Generated embedding for: '{text}' (dimension: {len(embedding_vector)})")
    return embedding_vector


def fetch_embeddings_batch(
        texts: list[str],
        *,
        client: AzureOpenAI,
        cache: dict[str, list[float]],
        model_name: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Generates embeddings for multiple texts in a single API call, with caching."""
    uncached_texts = [text for text in texts if text not in cache]

    if uncached_texts:
        cleaned_texts = [text.replace("\n", " ") for text in uncached_texts]
        response = client.embeddings.create(input=cleaned_texts, model=model_name)

        for i, uncached_text in enumerate(uncached_texts):
            cache[uncached_text] = response.data[i].embedding

    return [cache[text] for text in texts]


def compute_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculates cosine similarity between two embedding vectors."""
    return cosine_similarity([vector_a], [vector_b])[0][0]


def compute_euclidean_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculates Euclidean distance between two embedding vectors."""
    return float(np.linalg.norm(np.array(vector_a) - np.array(vector_b)))


def compare_two_words(
        word_a: str,
        word_b: str,
        *,
        client: AzureOpenAI,
        cache: dict[str, list[float]],
) -> dict[str, float]:
    """Compares two words using cosine similarity, euclidean distance, and dot product."""
    embedding_a = fetch_embedding(word_a, client=client, cache=cache)
    embedding_b = fetch_embedding(word_b, client=client, cache=cache)

    return {
        "cosine_similarity": float(compute_cosine_similarity(embedding_a, embedding_b)),
        "euclidean_distance": float(compute_euclidean_distance(embedding_a, embedding_b)),
        "dot_product": float(np.dot(embedding_a, embedding_b)),
    }


def demonstrate_basic_embedding_properties(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    Embeddings are high-dimensional numerical vectors that capture semantic meaning.

    Key properties:
    • Each word/phrase maps to a fixed-size vector (e.g., 1536 dimensions)
    • OpenAI normalizes embeddings to unit length (~1.0 magnitude)
    • Each dimension captures different semantic features
    • The statistics (min, max, mean, std) reveal the distribution of values
    """
    print_section_header("DEMO 1: BASIC EMBEDDINGS")
    print(textwrap.dedent(demonstrate_basic_embedding_properties.__doc__))

    sample_word = "cat"
    embedding_vector = fetch_embedding(sample_word, client=client, cache=cache)
    embedding_array = np.array(embedding_vector)

    print(textwrap.dedent(f"""\
        Word: '{sample_word}'
        Embedding dimensions: {len(embedding_vector)}
        First 5 dimensions: {embedding_vector[:5]}
        Embedding magnitude: {np.linalg.norm(embedding_array):.4f}

        Embedding statistics:
        Min value: {min(embedding_vector):.6f}
        Max value: {max(embedding_vector):.6f}
        Mean: {np.mean(embedding_array):.6f}
        Standard deviation: {np.std(embedding_array):.6f}"""))


def demonstrate_word_similarity_analysis(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    Similarity metrics measure how semantically related two words are.

    Three metrics compared:
    • Cosine Similarity: measures angle between vectors (higher = more similar)
    • Euclidean Distance: measures straight-line distance (lower = more similar)
    • Dot Product: magnitude-weighted similarity (correlates with similarity)
    """
    print_section_header("DEMO 2: WORD SIMILARITY ANALYSIS")
    print(textwrap.dedent(demonstrate_word_similarity_analysis.__doc__))

    word_pairs_to_compare: list[WordPair] = [
        WordPair(first="cat", second="dog"),
        WordPair(first="cat", second="kitten"),
        WordPair(first="dog", second="puppy"),
        WordPair(first="king", second="queen"),
        WordPair(first="man", second="woman"),
        WordPair(first="happy", second="joyful"),
        WordPair(first="car", second="automobile"),
        WordPair(first="bratwurst", second="sushi"),
    ]

    print(
        f"{'Word Pair':<20} {'Cosine Sim':>12} {'Euclidean':>12} {'Dot Product':>12}\n"
        f"{'-' * 58}"
    )

    for pair in word_pairs_to_compare:
        metrics = compare_two_words(pair.first, pair.second, client=client, cache=cache)
        print(
            f"{pair.first}-{pair.second:<15} {metrics['cosine_similarity']:>12.4f} "
            f"{metrics['euclidean_distance']:>12.4f} {metrics['dot_product']:>12.4f}"
        )

    print(textwrap.dedent("""\

        Key Observations:
        • Higher cosine similarity = more semantically similar
        • Lower euclidean distance = more similar
        • Dot product magnitude correlates with similarity"""))


def demonstrate_cat_dog_mystery(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    The Counterintuitive Finding: Why is 'cat' more similar to 'dog' than to 'kitten'?

    Embeddings capture statistical patterns of word usage, NOT biological taxonomy:
    • Co-occurrence in similar contexts matters most
    • 'cat' and 'dog' appear in nearly identical sentence structures
    • Semantic level relationships (both adult animals, pets) dominate
    • 'kitten' occupies a slightly different contextual niche (youth, cuteness)
    """
    print_section_header("DEMO 3: THE CAT-DOG MYSTERY")
    print(textwrap.dedent(demonstrate_cat_dog_mystery.__doc__))

    cat_dog_metrics = compare_two_words("cat", "dog", client=client, cache=cache)
    cat_kitten_metrics = compare_two_words("cat", "kitten", client=client, cache=cache)

    print(textwrap.dedent(f"""\
        The Counterintuitive Finding:
        cat-dog similarity:    {cat_dog_metrics['cosine_similarity']:.4f}
        cat-kitten similarity: {cat_kitten_metrics['cosine_similarity']:.4f}
        Difference: {cat_dog_metrics['cosine_similarity'] - cat_kitten_metrics['cosine_similarity']:.4f}"""))

    if cat_dog_metrics["cosine_similarity"] > cat_kitten_metrics["cosine_similarity"]:
        print(textwrap.dedent("""\

            🤔 Why is 'cat' more similar to 'dog' than to 'kitten'?

            This demonstrates that embeddings capture:
            • Statistical patterns of word usage
            • Co-occurrence in similar contexts
            • Semantic level relationships (adult animals)
            • NOT biological or taxonomic relationships"""))


def demonstrate_context_effects(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    Context dramatically changes how words relate to each other in vector space.

    Key insight: The same word pair can have reversed similarity rankings
    depending on the surrounding context. Phrases capture richer meaning
    than individual words, and context can completely flip relationships.
    """
    print_section_header("DEMO 4: CONTEXT MATTERS")
    print(textwrap.dedent(demonstrate_context_effects.__doc__))

    context_templates = [
        "The {} is sleeping peacefully",
        "I love my {} very much",
        "A {} playing with a toy",
        "Training a {} requires patience",
    ]

    context_test_words = ["cat", "dog", "kitten"]

    print(
        f"Similarity changes with context:\n"
        f"{'Context':<35} {'cat-dog':>10} {'cat-kitten':>12} {'Difference':>12}\n"
        f"{'-' * 70}"
    )

    for template in context_templates:
        phrases = [template.format(word) for word in context_test_words]
        phrase_embeddings = fetch_embeddings_batch(phrases, client=client, cache=cache)

        cat_phrase_embedding, dog_phrase_embedding, kitten_phrase_embedding = phrase_embeddings

        cat_dog_similarity = compute_cosine_similarity(cat_phrase_embedding, dog_phrase_embedding)
        cat_kitten_similarity = compute_cosine_similarity(cat_phrase_embedding, kitten_phrase_embedding)
        similarity_difference = cat_dog_similarity - cat_kitten_similarity

        truncated_context = template.replace(" {} ", " [X] ")[:30] + "..."
        print(
            f"{truncated_context:<35} {cat_dog_similarity:>10.4f} "
            f"{cat_kitten_similarity:>12.4f} {similarity_difference:>12.4f}"
        )

    print("\nKey Insight: Context can reverse similarity relationships!")


def demonstrate_semantic_clusters(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    Related concepts naturally cluster together in embedding space.

    Using PCA (Principal Component Analysis) we project high-dimensional
    embeddings into 2D for visualization. Words from the same category
    should appear close together, demonstrating that embeddings organize
    knowledge into meaningful semantic neighborhoods.
    """
    print_section_header("DEMO 5: SEMANTIC CLUSTERS VISUALIZATION")
    print(textwrap.dedent(demonstrate_semantic_clusters.__doc__))

    semantic_groups: dict[str, list[str]] = {
        "Animals": ["cat", "dog", "kitten", "puppy", "bird", "fish"],
        "Food": ["apple", "pizza", "sushi", "bread", "chocolate", "salad"],
        "Transport": ["car", "bicycle", "airplane", "train", "boat", "bus"],
        "Emotions": ["happy", "sad", "angry", "excited", "calm", "surprised"],
    }

    all_cluster_words: list[str] = []
    word_to_group_name: dict[str, str] = {}

    for group_name, words_in_group in semantic_groups.items():
        all_cluster_words.extend(words_in_group)
        for word in words_in_group:
            word_to_group_name[word] = group_name

    cluster_embeddings = fetch_embeddings_batch(all_cluster_words, client=client, cache=cache)
    embeddings_matrix = np.array(cluster_embeddings)

    pca_reducer = PCA(n_components=2)
    reduced_2d_embeddings = pca_reducer.fit_transform(embeddings_matrix)

    plt.figure(figsize=(12, 8))
    group_colors = ["red", "blue", "green", "orange"]

    for group_name, color in zip(semantic_groups.keys(), group_colors):
        group_word_indices = [
            idx for idx, word in enumerate(all_cluster_words)
            if word_to_group_name[word] == group_name
        ]
        group_x_coords = reduced_2d_embeddings[group_word_indices, 0]
        group_y_coords = reduced_2d_embeddings[group_word_indices, 1]

        plt.scatter(group_x_coords, group_y_coords, c=color, label=group_name, alpha=0.7, s=100)

        for word_idx in group_word_indices:
            plt.annotate(
                all_cluster_words[word_idx],
                (reduced_2d_embeddings[word_idx, 0], reduced_2d_embeddings[word_idx, 1]),
                fontsize=10,
                alpha=0.8,
            )

    plt.xlabel(f"First Principal Component (explains {pca_reducer.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Second Principal Component (explains {pca_reducer.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("Word Embeddings Projected to 2D Space")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("semantic_clusters.png", dpi=300, bbox_inches="tight")

    print(
        f"✅ Visualization saved as 'semantic_clusters.png'\n"
        f"📊 PCA explains {sum(pca_reducer.explained_variance_ratio_):.1%} of total variance"
    )

    intra_cluster_similarities: list[float] = []
    inter_cluster_similarities: list[float] = []

    for i in range(len(all_cluster_words)):
        for j in range(i + 1, len(all_cluster_words)):
            pairwise_similarity = compute_cosine_similarity(
                cluster_embeddings[i], cluster_embeddings[j]
            )

            if word_to_group_name[all_cluster_words[i]] == word_to_group_name[all_cluster_words[j]]:
                intra_cluster_similarities.append(pairwise_similarity)
            else:
                inter_cluster_similarities.append(pairwise_similarity)

    average_intra = np.mean(intra_cluster_similarities)
    average_inter = np.mean(inter_cluster_similarities)

    print(textwrap.dedent(f"""\

        Cluster Analysis:
        Average intra-cluster similarity: {average_intra:.4f}
        Average inter-cluster similarity: {average_inter:.4f}
        Clustering effectiveness: {average_intra - average_inter:.4f}"""))


def demonstrate_similarity_heatmap(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    A similarity heatmap reveals the full relationship matrix between words.

    By computing cosine similarity for every word pair, we can visualize
    which words are closest and farthest in embedding space. This helps
    identify unexpected relationships and validate semantic intuitions.
    """
    print_section_header("DEMO 6: SIMILARITY HEATMAP")
    print(textwrap.dedent(demonstrate_similarity_heatmap.__doc__))

    heatmap_words = ["cat", "kitten", "dog", "puppy", "animal", "pet", "feline", "canine"]
    heatmap_embeddings = fetch_embeddings_batch(heatmap_words, client=client, cache=cache)

    word_count = len(heatmap_words)
    similarity_matrix = np.zeros((word_count, word_count))

    for row_idx in range(word_count):
        for col_idx in range(word_count):
            similarity_matrix[row_idx, col_idx] = compute_cosine_similarity(
                heatmap_embeddings[row_idx], heatmap_embeddings[col_idx]
            )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        xticklabels=heatmap_words,
        yticklabels=heatmap_words,
        cmap="RdYlBu_r",
        center=0.5,
        fmt=".3f",
    )

    plt.title("Word Similarity Heatmap (Cosine Similarity)")
    plt.tight_layout()
    plt.savefig("similarity_heatmap.png", dpi=300, bbox_inches="tight")

    print("✅ Heatmap saved as 'similarity_heatmap.png'")

    np.fill_diagonal(similarity_matrix, -1)

    most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    least_similar_indices = np.unravel_index(np.argmin(similarity_matrix), similarity_matrix.shape)

    print(
        f"\nMost similar pair: {heatmap_words[most_similar_indices[0]]}-"
        f"{heatmap_words[most_similar_indices[1]]} "
        f"(similarity: {similarity_matrix[most_similar_indices]:.4f})\n"
        f"Least similar pair: {heatmap_words[least_similar_indices[0]]}-"
        f"{heatmap_words[least_similar_indices[1]]} "
        f"(similarity: {similarity_matrix[least_similar_indices]:.4f})"
    )


def compute_analogy_similarity(
        analogy: AnalogyTestCase,
        *,
        word_vectors: dict[str, np.ndarray],
        client: AzureOpenAI,
        cache: dict[str, list[float]],
) -> float:
    """Computes vector arithmetic for an analogy and returns cosine similarity to the expected target."""
    for word in [analogy.source_word, analogy.subtract_word, analogy.add_word, analogy.expected_target]:
        if word not in word_vectors:
            word_vectors[word] = np.array(fetch_embedding(word, client=client, cache=cache))

    analogy_result_vector = (
            word_vectors[analogy.source_word]
            - word_vectors[analogy.subtract_word]
            + word_vectors[analogy.add_word]
    )
    return compute_cosine_similarity(
        analogy_result_vector.tolist(), word_vectors[analogy.expected_target].tolist()
    )


def demonstrate_vector_arithmetic(client: AzureOpenAI, cache: dict[str, list[float]]) -> None:
    """
    Vector arithmetic reveals how embeddings encode semantic relationships geometrically.

    The famous analogy: king - man + woman ≈ queen
    • 'king' - 'man' isolates the concept of royalty without gender
    • Adding 'woman' reintroduces gender in the female direction
    • The result lands near 'queen' — royalty + female

    This works because relationships like gender, royalty, and family roles
    emerge as consistent directional offsets in embedding space.
    """
    print_section_header("DEMO 7: VECTOR ARITHMETIC - THE FAMOUS KING-QUEEN ANALOGY")
    print(textwrap.dedent(demonstrate_vector_arithmetic.__doc__))

    print(textwrap.dedent("""\
        🎯 Exploring the famous: king - man + woman ≈ queen
        This demonstrates how embeddings capture semantic relationships!

        1. Getting embeddings for the core analogy words..."""))
    core_analogy_words = ["king", "man", "woman", "queen"]
    word_vectors: dict[str, np.ndarray] = {}

    for word in core_analogy_words:
        word_vectors[word] = np.array(fetch_embedding(word, client=client, cache=cache))

    print(f"\n2. Performing vector arithmetic: king - man + woman")
    king_minus_man_plus_woman = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]
    print(f"✓ Calculated result vector (dimension: {len(king_minus_man_plus_woman)})")

    print(f"\n3. Comparing result to target word 'queen'...")
    queen_similarity_score = compute_cosine_similarity(
        king_minus_man_plus_woman.tolist(), word_vectors["queen"].tolist()
    )
    print(f"Direct similarity to 'queen': {queen_similarity_score:.4f}")

    print("\n4. Testing against candidate words...")
    candidate_words = ["queen", "princess", "lady", "woman", "king", "prince", "monarch", "ruler", "empress"]
    candidate_similarities: list[tuple[str, float]] = []

    for candidate in candidate_words:
        if candidate not in word_vectors:
            word_vectors[candidate] = np.array(fetch_embedding(candidate, client=client, cache=cache))

        candidate_score = compute_cosine_similarity(
            king_minus_man_plus_woman.tolist(), word_vectors[candidate].tolist()
        )
        candidate_similarities.append((candidate, candidate_score))

    candidate_similarities.sort(key=lambda pair: pair[1], reverse=True)

    print(
        f"\n{'Word':<12} {'Similarity':>12} {'Status'}\n"
        f"{'-' * 35}"
    )

    queen_rank: int | None = None
    for rank_index, (candidate_word, similarity_score) in enumerate(candidate_similarities):
        if candidate_word == "queen":
            status = "🎯 TARGET WORD"
            queen_rank = rank_index + 1
        else:
            status = ""
        print(f"{candidate_word:<12} {similarity_score:>12.4f} {status}")

    print(f"\nResult: 'queen' ranked #{queen_rank} out of {len(candidate_similarities)} candidates")

    if queen_rank == 1:
        print(textwrap.dedent("""\
            ✅ PERFECT: The analogy works flawlessly!
               king - man + woman ≈ queen"""))
    elif queen_rank is not None and queen_rank <= 3:
        print("✅ EXCELLENT: The analogy works very well (top 3 result)")
    else:
        print("✅ GOOD: The analogy demonstrates the pattern")

    print(textwrap.dedent("""\

        5. Understanding why this works...
        The vector arithmetic captures relationships:
        • 'king' - 'man' = concept of royalty without gender
        • Adding 'woman' = royalty + female gender
        • Result ≈ 'queen' = royal female"""))

    print(f"\n6. Testing other gender-role analogies...")
    gender_role_analogies: list[AnalogyTestCase] = [
        AnalogyTestCase(source_word="father", subtract_word="man", add_word="woman", expected_target="mother"),
        AnalogyTestCase(source_word="uncle", subtract_word="man", add_word="woman", expected_target="aunt"),
        AnalogyTestCase(source_word="boy", subtract_word="male", add_word="female", expected_target="girl"),
        AnalogyTestCase(source_word="prince", subtract_word="man", add_word="woman", expected_target="princess"),
    ]

    print(
        f"\n{'Analogy':<25} {'Target':<8} {'Similarity':>12} {'Success':>10}\n"
        f"{'-' * 57}"
    )

    for analogy in gender_role_analogies:
        analogy_similarity = compute_analogy_similarity(
            analogy, word_vectors=word_vectors, client=client, cache=cache,
        )

        success_indicator = "✅" if analogy_similarity > 0.6 else "⚠️" if analogy_similarity > 0.4 else "❌"
        analogy_description = f"{analogy.source_word} - {analogy.subtract_word} + {analogy.add_word}"
        print(
            f"{analogy_description:<25} {analogy.expected_target:<8} "
            f"{analogy_similarity:>12.4f} {success_indicator:>10}"
        )

    print(f"\n7. Testing conceptual analogies...")
    conceptual_analogies: list[AnalogyTestCase] = [
        AnalogyTestCase(source_word="Paris", subtract_word="France", add_word="Italy", expected_target="Rome"),
        AnalogyTestCase(source_word="big", subtract_word="bigger", add_word="small", expected_target="smaller"),
        AnalogyTestCase(source_word="walk", subtract_word="walking", add_word="run", expected_target="running"),
    ]

    print(
        f"\n{'Conceptual Analogy':<25} {'Target':<10} {'Similarity':>12}\n"
        f"{'-' * 49}"
    )

    for analogy in conceptual_analogies:
        analogy_similarity = compute_analogy_similarity(
            analogy, word_vectors=word_vectors, client=client, cache=cache,
        )

        analogy_description = f"{analogy.source_word} - {analogy.subtract_word} + {analogy.add_word}"
        print(f"{analogy_description:<25} {analogy.expected_target:<10} {analogy_similarity:>12.4f}")


def print_learning_summary() -> None:
    print_section_header("🎉 COMPREHENSIVE EMBEDDINGS DEMO COMPLETE!")

    print(textwrap.dedent("""\
        🔍 KEY INSIGHTS FROM OUR EXPLORATION:

        1. 📊 BASIC PROPERTIES:
           • Embeddings are high-dimensional vectors (1536 dimensions)
           • OpenAI normalizes embeddings to unit length (~1.0 magnitude)
           • Each dimension captures different semantic features

        2. 🔗 SIMILARITY PATTERNS:
           • Cosine similarity measures semantic relationships
           • Counter-intuitive results: cat-dog > cat-kitten
           • Statistical co-occurrence matters more than logical relationships

        3. 🎭 CONTEXT IS EVERYTHING:
           • Same words in different contexts have different embeddings
           • Phrases capture richer meaning than individual words
           • Context can completely reverse similarity relationships

        4. 🗂️ SEMANTIC CLUSTERING:
           • Related concepts naturally cluster together
           • PCA visualization reveals meaningful word groupings
           • Intra-cluster similarity > inter-cluster similarity

        5. 🧮 VECTOR ARITHMETIC MAGIC:
           • king - man + woman ≈ queen (geometric relationships)
           • Gender, royalty, family roles emerge as 'directions'
           • Analogical reasoning through mathematical operations
           • Success varies by relationship type and embedding quality

        6. 🏗️ FOUNDATION FOR AI:
           • Embeddings power modern LLMs, search, and recommendation systems
           • They enable machines to understand semantic relationships
           • Vector databases use these properties for similarity search

        🎯 PRACTICAL APPLICATIONS:
        • Search engines (semantic search)
        • Recommendation systems (content similarity)
        • Language models (understanding context)
        • Clustering and classification tasks
        • Question answering and chatbots

        ✨ The magic of embeddings: turning words into geometry!
           Every word becomes a point in space where distance = meaning
    """))


if __name__ == "__main__":
    load_dotenv(override=True)
    azure_openai_client = AzureOpenAI()
    embedding_cache: dict[str, list[float]] = {}

    demonstrate_basic_embedding_properties(azure_openai_client, embedding_cache)
    demonstrate_word_similarity_analysis(azure_openai_client, embedding_cache)
    demonstrate_cat_dog_mystery(azure_openai_client, embedding_cache)
    demonstrate_context_effects(azure_openai_client, embedding_cache)
    demonstrate_semantic_clusters(azure_openai_client, embedding_cache)
    demonstrate_similarity_heatmap(azure_openai_client, embedding_cache)
    demonstrate_vector_arithmetic(azure_openai_client, embedding_cache)
    print_learning_summary()
