import gradio as gr
from fact_check_llm import verify_news, verify_news_audio
from image_fact_checker import verify_image_news


###Start of code to handle web interface via GridIO
input_audio = gr.Audio(
    sources=["microphone"],
    type='filepath',
    format = 'mp3',
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

def handle_input(choice, text_input, audio_input, image_input):
    """
     Description: Based on user selection decides the mode of input and corresponding calling function
    """
    if choice == 'Text':
        if text_input.strip() == '':
            return 'Please input the news to verify','Please input the news to verify','Please input the news to verify',''
        return verify_news(text_input) + ('',)
    elif choice == 'Audio':
        if audio_input:
            result = verify_news_audio(audio_input)
            return result + ('',)
        else:
            return 'Please record something first','Please record something first','Please record something first',''
    elif choice == 'Image':
        if image_input:
            return verify_image_news(image_input)
        else:
            return 'Please upload an image first','Please upload an image first','Please upload an image first',''
    else:
        return 'Please chose one of the radio button for input','Please chhose one of the radio button for input','Please chhose one of the radio button for input',''
        
def clear_all():
    """
       Clears all the input/output box in the screen
    """
    return (
            "",                  # text_box
            None,                # audio_rec
            None,                # image_input
            "",                  # output1
            "",                  # output2
            "",                  # output3
            ""                   # output_ocr
           )

with gr.Blocks() as demo:
    choice = gr.Radio(["Text", "Audio", "Image"], label="Select Your Input Method", value="Text")

    with gr.Column():
        text_box = gr.Textbox(label="Enter the news to verify (in any language)", visible=True)
        audio_rec = gr.Audio(sources="microphone", type="filepath", label="Record the news to verify (in any language)", visible=False)
        image_input = gr.Image(label="Upload news image (screenshot, photo, etc.)", type="filepath", visible=False)

        submit_btn = gr.Button("Submit")

        output1 = gr.Textbox(label="Claim headline (in English)")
        output2 = gr.Textbox(label="Verification Result (in English)")
        output3 = gr.Textbox(label="Verification Result (in original input language)")
        output_ocr = gr.Textbox(label="Extracted Text from Image (OCR)", visible=True)
        
        clear_btn = gr.Button("Clear")

    def toggle_visibility(selected):
        return {
            text_box: gr.update(visible=(selected == "Text")),
            audio_rec: gr.update(visible=(selected == "Audio")),
            image_input: gr.update(visible=(selected == "Image")),
            output1: gr.update(value=""),
            output2: gr.update(value=""),
            output3: gr.update(value=""),
            output_ocr: gr.update(value="")
        }

    choice.change(fn=toggle_visibility, inputs=choice, outputs=[text_box, audio_rec, image_input, output1, output2, output3, output_ocr])

    submit_btn.click(fn=handle_input, inputs=[choice, text_box, audio_rec, image_input], outputs = [output1, output2, output3, output_ocr])
    clear_btn.click(fn=clear_all, inputs=[], outputs=[text_box, audio_rec, image_input, output1, output2, output3, output_ocr]
    )

###End of code to handle web interface via GridIO

demo.launch()

