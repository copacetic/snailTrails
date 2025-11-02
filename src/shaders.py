"""
Shader loading and compilation
"""

import os
from typing import Dict
import moderngl


class ShaderManager:
    """Manages shader loading and compilation"""

    def __init__(self, ctx: moderngl.Context, shader_dir: str = 'shaders'):
        """
        Initialize shader manager

        Args:
            ctx: ModernGL context
            shader_dir: Directory containing shader files
        """
        self.ctx = ctx
        self.shader_dir = shader_dir
        self.shaders = {}

    def load_shader_source(self, filename: str) -> str:
        """
        Load shader source from file

        Args:
            filename: Shader filename

        Returns:
            Shader source code

        Raises:
            FileNotFoundError: If shader file doesn't exist
        """
        filepath = os.path.join(self.shader_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Shader file not found: {filepath}")

        with open(filepath, 'r') as f:
            return f.read()

    def compile_compute_shader(self, filename: str, name: str = None) -> moderngl.ComputeShader:
        """
        Compile compute shader

        Args:
            filename: Shader filename
            name: Optional name to cache shader

        Returns:
            Compiled compute shader

        Raises:
            Exception: If shader compilation fails
        """
        source = self.load_shader_source(filename)

        try:
            shader = self.ctx.compute_shader(source)
            if name:
                self.shaders[name] = shader
            return shader
        except Exception as e:
            raise Exception(f"Failed to compile compute shader {filename}: {e}")

    def compile_render_program(
        self,
        vertex_file: str,
        fragment_file: str,
        name: str = None
    ) -> moderngl.Program:
        """
        Compile vertex/fragment shader program

        Args:
            vertex_file: Vertex shader filename
            fragment_file: Fragment shader filename
            name: Optional name to cache program

        Returns:
            Compiled shader program

        Raises:
            Exception: If shader compilation fails
        """
        vertex_source = self.load_shader_source(vertex_file)
        fragment_source = self.load_shader_source(fragment_file)

        try:
            program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source
            )
            if name:
                self.shaders[name] = program
            return program
        except Exception as e:
            raise Exception(
                f"Failed to compile shader program "
                f"({vertex_file}, {fragment_file}): {e}"
            )

    def get_shader(self, name: str):
        """Get cached shader by name"""
        return self.shaders.get(name)

    def validate_shader_files(self) -> Dict[str, bool]:
        """
        Validate that all required shader files exist

        Returns:
            Dict mapping shader names to existence status
        """
        required_shaders = [
            'field_compute.glsl',
            'agent_compute.glsl',
            'vertex.glsl',
            'fragment.glsl'
        ]

        results = {}
        for shader in required_shaders:
            filepath = os.path.join(self.shader_dir, shader)
            results[shader] = os.path.exists(filepath)

        return results

    def list_available_shaders(self) -> list:
        """List all .glsl files in shader directory"""
        if not os.path.exists(self.shader_dir):
            return []

        return [
            f for f in os.listdir(self.shader_dir)
            if f.endswith('.glsl')
        ]
