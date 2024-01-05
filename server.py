from flask import Flask, jsonify, request
from flask_cors import CORS

import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

app = Flask(__name__)
CORS(app, origins=["http://your-allowed-domain.com", "http://127.0.0.1:5173"])


# Load shape-e model and diffusion configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

def transform_text_to_mesh(prompt):
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

    meshes = []
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm, latent).tri_mesh()
        meshes.append(mesh)
    return meshes

@app.route('/generate_mesh', methods=['POST'])
def generate_mesh():
    if request.method == 'POST':

        data = request.json
        prompt = data['text']

        meshes = transform_text_to_mesh(prompt)

        file_paths = []
        for i, mesh in enumerate(meshes):
            file_path = f'car_{i}.obj'
            with open(file_path, 'w') as f:
                mesh.write_obj(f)
            file_paths.append(file_path)

        return jsonify({'message': 'Mesh generated successfully', 'file_paths': file_paths})

if __name__ == '__main__':
    app.run()
