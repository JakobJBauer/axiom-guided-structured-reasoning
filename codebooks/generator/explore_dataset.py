from datasets import load_dataset
from collections import Counter
import json

def explore_dataset(sample_size=50000):
    ds = load_dataset("SimpleStories/SimpleStories", split="train")
    sample_size = min(sample_size, len(ds))
    
    print(f"Analyzing {sample_size} examples from {len(ds)} total examples...\n")
    
    # Collect unique values
    categorical_features = {
        'topic': [],
        'theme': [],
        'style': [],
        'feature': [],
        'grammar': [],
        'persona': [],
        'initial_word_type': [],
        'initial_letter': []
    }
    
    numeric_features = {
        'word_count': [],
        'character_count': [],
        'num_paragraphs': [],
        'avg_word_length': [],
        'avg_sentence_length': [],
        'flesch_reading_ease': [],
        'flesch_kincaid_grade': [],
        'dale_chall_readability_score': []
    }
    
    for i in range(sample_size):
        example = ds[i]
        for feat in categorical_features:
            val = example.get(feat)
            if val and str(val).strip():
                categorical_features[feat].append(str(val).strip())
        
        for feat in numeric_features:
            val = example.get(feat)
            if val is not None:
                numeric_features[feat].append(val)
    
    # Get most common values
    print("Most common categorical values:")
    for feat, values in categorical_features.items():
        counter = Counter(values)
        print(f"\n{feat} ({len(counter)} unique):")
        for val, count in counter.most_common(10):
            print(f"  {val}: {count}")
    
    # Calculate thresholds for numeric features
    print("\n\nNumeric feature statistics:")
    for feat, values in numeric_features.items():
        if values:
            sorted_vals = sorted(values)
            q25 = sorted_vals[len(sorted_vals)//4]
            q50 = sorted_vals[len(sorted_vals)//2]
            q75 = sorted_vals[3*len(sorted_vals)//4]
            print(f"\n{feat}:")
            print(f"  min={min(values):.2f}, q25={q25:.2f}, median={q50:.2f}, q75={q75:.2f}, max={max(values):.2f}")
    
    return categorical_features, numeric_features


def propose_leaf_nodes(categorical_features, numeric_features):
    leaf_nodes = []
    
    # 1. Topic-based nodes (10 nodes) - most common topics
    topic_counter = Counter(categorical_features['topic'])
    top_topics = [topic for topic, _ in topic_counter.most_common(10)]
    for topic in top_topics:
        # Create a clean node name
        node_name = topic.lower().replace(' ', '-').replace(',', '').replace("'", "")
        node_name = f"topic-{node_name[:30]}"  # Limit length
        leaf_nodes.append({
            'id': node_name,
            'description': f"The story is about {topic}",
            'type': 'categorical_match',
            'feature': 'topic',
            'value': topic
        })
    
    # 2. Theme-based nodes (10 nodes)
    theme_counter = Counter(categorical_features['theme'])
    top_themes = [theme for theme, _ in theme_counter.most_common(10)]
    for theme in top_themes:
        node_name = theme.lower().replace(' ', '-').replace(',', '')
        node_name = f"theme-{node_name[:30]}"
        leaf_nodes.append({
            'id': node_name,
            'description': f"The story has the theme {theme}",
            'type': 'categorical_match',
            'feature': 'theme',
            'value': theme
        })
    
    # 3. Style-based nodes (8 nodes)
    style_counter = Counter(categorical_features['style'])
    top_styles = [style for style, _ in style_counter.most_common(8)]
    for style in top_styles:
        node_name = style.lower().replace(' ', '-').replace('-', '-')
        node_name = f"style-{node_name}"
        leaf_nodes.append({
            'id': node_name,
            'description': f"The story is {style}",
            'type': 'categorical_match',
            'feature': 'style',
            'value': style
        })
    
    # 4. Feature-based nodes (8 nodes) - story features
    feature_counter = Counter(categorical_features['feature'])
    top_features = [feat for feat, _ in feature_counter.most_common(8)]
    for feat in top_features:
        node_name = feat.lower().replace(' ', '-').replace("'", "").replace("'s", "")
        node_name = f"has-{node_name[:25]}"
        leaf_nodes.append({
            'id': node_name,
            'description': f"The story has {feat}",
            'type': 'categorical_match',
            'feature': 'feature',
            'value': feat
        })
    
    # 5. Grammar-based nodes (5 nodes)
    grammar_counter = Counter(categorical_features['grammar'])
    top_grammars = [gram for gram, _ in grammar_counter.most_common(5)]
    for gram in top_grammars:
        node_name = gram.lower().replace(' ', '-')
        node_name = f"uses-{node_name[:25]}"
        leaf_nodes.append({
            'id': node_name,
            'description': f"The story uses {gram}",
            'type': 'categorical_match',
            'feature': 'grammar',
            'value': gram
        })
    
    # 6. Initial word type nodes (4 nodes)
    for word_type in ['noun', 'adjective', 'adverb', 'preposition']:
        leaf_nodes.append({
            'id': f"starts-with-{word_type}",
            'description': f"The story starts with a {word_type}",
            'type': 'categorical_match',
            'feature': 'initial_word_type',
            'value': word_type
        })
    
    # 7. Numeric threshold nodes (5 nodes)
    word_counts = sorted(numeric_features['word_count'])
    if word_counts:
        thresholds = {
            'short': word_counts[len(word_counts)//4],  # Q1
            'medium': word_counts[len(word_counts)//2],  # Median
            'long': word_counts[3*len(word_counts)//4],  # Q3
        }
        
        leaf_nodes.append({
            'id': 'short-story',
            'description': f"The story has fewer than {int(thresholds['short'])} words",
            'type': 'numeric_threshold',
            'feature': 'word_count',
            'threshold': thresholds['short'],
            'operator': '<'
        })
        
        leaf_nodes.append({
            'id': 'long-story',
            'description': f"The story has more than {int(thresholds['long'])} words",
            'type': 'numeric_threshold',
            'feature': 'word_count',
            'threshold': thresholds['long'],
            'operator': '>'
        })
    
    # Add readability nodes
    readability_scores = sorted(numeric_features['flesch_reading_ease'])
    if readability_scores:
        low_threshold = readability_scores[len(readability_scores)//4]
        high_threshold = readability_scores[3*len(readability_scores)//4]
        
        leaf_nodes.append({
            'id': 'low-readability',
            'description': f"The story has readability score below {low_threshold:.1f}",
            'type': 'numeric_threshold',
            'feature': 'flesch_reading_ease',
            'threshold': low_threshold,
            'operator': '<'
        })
        
        leaf_nodes.append({
            'id': 'high-readability',
            'description': f"The story has readability score above {high_threshold:.1f}",
            'type': 'numeric_threshold',
            'feature': 'flesch_reading_ease',
            'threshold': high_threshold,
            'operator': '>'
        })
    
    # Add paragraph count node
    paragraphs = sorted(numeric_features['num_paragraphs'])
    if paragraphs:
        many_paragraphs = paragraphs[3*len(paragraphs)//4]
        leaf_nodes.append({
            'id': 'many-paragraphs',
            'description': f"The story has more than {int(many_paragraphs)} paragraphs",
            'type': 'numeric_threshold',
            'feature': 'num_paragraphs',
            'threshold': many_paragraphs,
            'operator': '>'
        })
    
    return leaf_nodes[:50]  # Return exactly 50


if __name__ == '__main__':
    categorical_features, numeric_features = explore_dataset()
    
    print("\n" + "="*80)
    print("PROPOSED 50 LEAF NODES")
    print("="*80 + "\n")
    
    leaf_nodes = propose_leaf_nodes(categorical_features, numeric_features)
    
    print(f"Generated {len(leaf_nodes)} leaf nodes:\n")
    for i, node in enumerate(leaf_nodes, 1):
        print(f"{i:2d}. {node['id']}")
        print(f"    {node['description']}")
        print()
    
    # Save to JSON
    output_file = 'proposed_leaf_nodes.json'
    with open(output_file, 'w') as f:
        json.dump(leaf_nodes, f, indent=2)
    print(f"\nSaved to {output_file}")

