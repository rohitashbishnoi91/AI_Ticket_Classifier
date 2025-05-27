import gradio as gr
from pipeline import TicketClassifier

# Initialize classifier
classifier = TicketClassifier()

def predict_ticket(text):
    """
    Predicts the issue type, urgency level, and entities from the given ticket text.

    Args:
        text (str): The input text of the ticket to be classified.

    Returns:
        tuple: A tuple containing:
            - issue_type (str): The predicted type of issue.
            - urgency_level (str): The predicted urgency level.
            - entities (Any): The extracted entities from the ticket text.
    """
    result = classifier.predict(text)
    return result['issue_type'], result['urgency_level'], result['entities']

iface = gr.Interface(
    fn=predict_ticket,
    inputs=gr.Textbox(lines=4, label="Enter Ticket Text"),
    outputs=[
        gr.Label(label="Issue Type"),
        gr.Label(label="Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="AI Ticket Classifier",
    examples=[
        ["My printer is broken and won't print anything"],
        ["Router connection keeps dropping every few minutes"]
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)  