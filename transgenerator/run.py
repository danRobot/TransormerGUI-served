import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)

class Generator():
    def __init__(self,args):
        self.args=args
        self.set_seed(self.args.seed)
        self.logger = logging.getLogger(__name__)
        self.MAX_LENGTH = int(10000)
        self.MODEL_CLASSES = {
                                "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
                                "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
                                "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
                                "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
                                "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
                                "xlm": (XLMWithLMHeadModel, XLMTokenizer),
                            }
        self.PREPROCESSING_FUNCTIONS = {
                                "ctrl": self.prepare_ctrl_input,
                                "xlm": self.prepare_xlm_input,
                                "xlnet": self.prepare_xlnet_input,
                                "transfo-xl": self.prepare_transfoxl_input,
                            }
        self.PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
                        (except for Alexei and Maria) are discovered.
                        The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
                        remainder of the story. 1883 Western Siberia,
                        a young Grigori Rasputin is asked by his father and a group of men to perform magic.
                        Rasputin has a vision and denounces one of the men as a horse thief. Although his
                        father initially slaps him for making such an accusation, Rasputin watches as the
                        man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
                        the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
                        with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

        self.args.n_gpu = torch.cuda.device_count()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        try:
            self.args.model_type = self.args.model_type.lower()
            model_class, tokenizer_class = self.MODEL_CLASSES[self.args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,cache_dir=args.wpath)
        self.model = model_class.from_pretrained(args.model_name_or_path,cache_dir=args.wpath)
        self.model.to(self.device)

        self.args.length = self.adjust_length_to_model(args.length, max_sequence_length=self.model.config.max_position_embeddings)
        self.logger.info(self.args)
        pass
    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def prepare_ctrl_input(self, prompt_text):
        if self.args.temperature > 0.7:
            self.logger.info("CTRL typically works better with lower temperatures (and lower top_k).")
        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not any(encoded_prompt[0] == x for x in self.tokenizer.control_codes.values()):
            self.logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
        return prompt_text
    
    def prepare_xlm_input(self, prompt_text):
        # kwargs = {"language": None, "mask_token_id": None}

        # Set the language
        use_lang_emb = hasattr(self.model.config, "use_lang_emb") and self.model.config.use_lang_emb
        if hasattr(self.model.config, "lang2id") and use_lang_emb:
            available_languages = self.model.config.lang2id.keys()
            if self.args.xlm_language in available_languages:
                language = self.args.xlm_language
            else:
                language = None
                while language not in available_languages:
                    language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

            self.model.config.lang_id = self.model.config.lang2id[language]
            # kwargs["language"] = tokenizer.lang2id[language]

        # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
        # XLM masked-language modeling (MLM) models need masked token
        # is_xlm_mlm = "mlm" in args.model_name_or_path
        # if is_xlm_mlm:
        #     kwargs["mask_token_id"] = tokenizer.mask_token_id

        return prompt_text
    def prepare_xlnet_input(self, prompt_text):
        prompt_text = (self.args.padding_text if self.args.padding_text else self.PADDING_TEXT) + prompt_text
        return prompt_text


    def prepare_transfoxl_input(self, prompt_text):
        prompt_text = (self.args.padding_text if self.args.padding_text else self.PADDING_TEXT) + prompt_text
        return prompt_text
    
    def adjust_length_to_model(self,length,max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = self.MAX_LENGTH  # avoid infinite loop
        return length

    
    def generate(self,prompt_text,
                      temperature,
                      top_k,
                      top_p):
        '''
        if(temperature==0):
            temperature=self.args.temperature
        if(top_k==0):
            top_k=self.args.k
        if(top_p==0):
            top_k=self.args.p
        '''
        requires_preprocessing = self.args.model_type in self.PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = self.PREPROCESSING_FUNCTIONS.get(self.args.model_type)
            preprocessed_prompt_text = prepare_input(self.args, self.model, self.tokenizer, prompt_text)
            encoded_prompt = self.tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
            )
        else:
            encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
                            input_ids=encoded_prompt,
                            max_length=self.args.length + len(encoded_prompt[0]),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=self.args.repetition_penalty,
                            do_sample=True,
                            num_return_sequences=self.args.num_return_sequences,
                        )
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            #print(total_sequence)
        return generated_sequences
    
