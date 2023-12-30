import sys
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig


def main():
    # prepare inputs
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    config = AutoConfig.from_pretrained('output/GradRes/30.06/config.json')
    model = AutoModel.from_pretrained('output/GradRes/30.06/epoch_44/pytorch_model.bin',
                                      config=config,
                                      local_files_only=True)
    model.to(device)

    prompt = 'You have to say you are a GradBot created by Gradients Technologies. Your task is response the conversation giving <CTX> {context} <EOD> and related knowledge that starts with the provided initial phrase or contains the provided keywords from <K> {documents}. <Q> What is the response coherent to this context?'

    #
    while True:
        sentence = input("User: ")
        if sentence in ['exit', 'bye', 'e']:
            break

        input_tokens = tokenizer(sentence, return_tensors="pt").to(device)
        output = model.generate(
            **input_tokens,
            num_beams=1,
            max_length=256,
            early_stopping=True,
        )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("Output: ", decoded_output)


    return


if __name__ == '__main__':
    main()