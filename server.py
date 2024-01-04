from flask import Flask, request, jsonify
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

app = Flask(__name__)

# 디바이스 체크가 필요
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

@app.route('/generate_model', methods=['POST'])
def generate_model():
    try:
        data = request.json
        prompt = data['text']

        batch_size = 1
        guidance_scale = 15.0

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1E-3,
            sigma_max=160,
            s_churn=0,
        )

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(xm, latent).tri_mesh()
            with open(f'example_mesh_{i}.ply', 'wb') as f:
                t.write_ply(f)
            with open(f'example_mesh_{i}.obj', 'w') as f:
                t.write_obj(f)

        return jsonify({'message': 'Model generated successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
