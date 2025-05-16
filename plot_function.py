import matplotlib.pyplot as plt

def plot_chatcount(df_data):
    chat_counts = df_data['Name'].value_counts()
    fig_height = max(2, len(chat_counts) * 0.6)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Set figure background color (outside plotting area)
    fig.patch.set_facecolor('#ADD8E6')  # light blue hex code

    # Set axes background color (inside plotting area)
    ax.set_facecolor('#ADD8E6')

    ax.axis('off')

    # Reverse order so first name is at the top
    names_counts = list(chat_counts.items())[::-1]

    for i, (name, count) in enumerate(names_counts):
        y_pos = i + 1
        ax.text(0.01, y_pos, f"$\\bf{{{name}}}$: {count} chats", fontsize=12, va='bottom')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, len(names_counts) + 1.5)

    ax.set_title("Number of Chats per Name", fontsize=14, weight='bold', loc='left', pad=20)

    fig.tight_layout()
    return fig



def plot_piechart(chat_emotions, label_map):
    # Define custom colors for each emotion label
    label_color_map = {
        'Sadness':  (0, 1, 1, 0.8),  # (r,g,b, alpha)
        'Joy':      (1, 1, 0, 1),
        'Love':    (1.0, 0.078, 0.576, 0.8),
        'Anger':   (214/255, 39/255, 40/255, 0.8),
        'Fear':   (31/255, 119/255, 180/255, 0.8),
        'Surprise':(0.565, 0.933, 0.565, 0.8) 
    }

    num_users = len(chat_emotions)
    cols = min(3, num_users)
    rows = (num_users + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if num_users > 1 else [axes]

    for idx, (name, preds) in enumerate(chat_emotions.items()):
        counts = preds.value_counts().sort_index()
        labels = [label_map.get(label, str(label)) for label in counts.index]

        # Get the colors corresponding to the labels
        colors = [label_color_map[label] for label in labels if label in label_color_map]

        # Ensure the number of colors matches the number of labels
        if len(colors) < len(labels):
            colors += ['#333333'] * (len(labels) - len(colors))  # Fill missing with gray

        axes[idx].pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops={'fontsize': 12}
        )
        axes[idx].set_title(f'{name}', fontsize=12, weight='bold')

    # Hide any extra axes
    for j in range(len(chat_emotions), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Chat Emotion Analysis per User', fontsize=14, color='red', weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    return fig  # Return fig so you can use st.pyplot(fig) in Streamlit



