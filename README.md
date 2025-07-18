# NEURAL-STYLE-TRANSFER

*COMPANY* : CODETECH IT SOLUTIONS

*NAME* :  NAGISETTY HEMANTH SAI

*INTERN ID* : CT04DG2507

*DOMAIN* : AI(ARTIFICIAL INTELLIGENCE)

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

# DESCRIPTION OF THE PROJECT

Neural Style Transfer (NST) is a fascinating application of deep learning that allows us to blend two images together — one for the content and another for the style. It means you can take a regular photo and give it the artistic flavor of famous paintings like Van Gogh’s Starry Night or Picasso’s Cubism. This creative process doesn't involve manually editing images. Instead, it relies on a neural network, particularly a pre-trained Convolutional Neural Network (CNN), to do the work.

The idea behind NST is to extract two sets of features from images: content features and style features. The content image is usually a photo (say, a building or a landscape), and the style image is a painting or any piece of artwork. Using a deep neural network like VGG19, we can isolate the visual patterns that define content (shapes, structure) and style (textures, brushstrokes, colors). The algorithm then generates a new image that combines the structure of the content image with the artistic touch of the style image.

The implementation leverages PyTorch, a popular deep learning library. VGG19, which has been trained on the ImageNet dataset, is used to extract the necessary features. It’s not trained from scratch — we only use it to obtain intermediate outputs from specific layers. These layers are chosen because they respond differently to content and style.

The training process isn’t like regular deep learning where the model learns weights. Instead, we start with a copy of the content image and treat it as a variable. The goal is to iteratively adjust this image so that its features match the content of the original photo and the style of the artwork. This is done by calculating two losses: content loss and style loss. Content loss ensures the structure remains, while style loss focuses on textures and patterns. A total loss combines both, and backpropagation is used to tweak the pixels of the generated image.

Once the optimization is complete, the result is a stylized image that reflects the desired look. This process is flexible. You can try different styles, adjust the weight given to content vs. style, and get drastically different results.

Neural Style Transfer has grown beyond an academic experiment. It’s used in creative apps, video filters, and design tools. It brings the power of AI into the hands of everyday users, making it possible to turn any snapshot into something that looks hand-painted.

Overall, NST shows how deep learning isn't just about automation — it's also about enabling creativity in entirely new ways. With just a few lines of code, we can build systems that make art, transforming ordinary visuals into something extraordinary.

# OUTPUT

![Image](https://github.com/user-attachments/assets/f7a95107-4366-42e2-b4e0-df5e899779cf)


