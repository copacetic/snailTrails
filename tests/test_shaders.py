"""
Tests for shader management
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shaders import ShaderManager


class TestShaderManager:
    """Test shader management (without GPU context)"""

    def test_validate_shader_files(self):
        """Test that all required shader files exist"""
        # Note: ShaderManager requires a context, but we can test file validation
        # by checking the shader directory directly

        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')
        required_files = [
            'field_compute.glsl',
            'agent_compute.glsl',
            'vertex.glsl',
            'fragment.glsl'
        ]

        for filename in required_files:
            filepath = os.path.join(shader_dir, filename)
            assert os.path.exists(filepath), f"Missing shader file: {filename}"

    def test_shader_file_contents(self):
        """Test that shader files have valid GLSL content"""
        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')

        # Field compute shader
        with open(os.path.join(shader_dir, 'field_compute.glsl'), 'r') as f:
            content = f.read()
            assert '#version 430' in content
            assert 'layout(local_size_x = 16, local_size_y = 16)' in content
            assert 'buffer VectorField' in content
            assert 'uniform int gridSize' in content
            assert 'uniform int samples' in content

        # Agent compute shader
        with open(os.path.join(shader_dir, 'agent_compute.glsl'), 'r') as f:
            content = f.read()
            assert '#version 430' in content
            assert 'layout(local_size_x = 512)' in content  # EXTREME mode
            assert 'struct Agent' in content
            assert 'atomicCompSwap' in content

        # Vertex shader
        with open(os.path.join(shader_dir, 'vertex.glsl'), 'r') as f:
            content = f.read()
            assert '#version 430' in content
            assert 'in vec2 in_position' in content
            assert 'uniform mat4 projection' in content

        # Fragment shader
        with open(os.path.join(shader_dir, 'fragment.glsl'), 'r') as f:
            content = f.read()
            assert '#version 430' in content
            assert 'out vec4 fragColor' in content

    def test_shader_syntax_basics(self):
        """Test basic shader syntax validity"""
        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')

        for shader_file in os.listdir(shader_dir):
            if shader_file.endswith('.glsl'):
                filepath = os.path.join(shader_dir, shader_file)
                with open(filepath, 'r') as f:
                    content = f.read()

                    # Should have version directive
                    assert '#version' in content, f"{shader_file} missing version directive"

                    # Should have main function
                    assert 'void main()' in content, f"{shader_file} missing main function"

                    # Should not have obvious syntax errors
                    assert '})' not in content, f"{shader_file} has mismatched braces"

    def test_compute_shader_bindings(self):
        """Test that compute shaders have correct buffer bindings"""
        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')

        # Field compute should bind buffer 0
        with open(os.path.join(shader_dir, 'field_compute.glsl'), 'r') as f:
            content = f.read()
            assert 'binding = 0' in content

        # Agent compute should bind buffers 0-3
        with open(os.path.join(shader_dir, 'agent_compute.glsl'), 'r') as f:
            content = f.read()
            assert 'binding = 0' in content
            assert 'binding = 1' in content
            assert 'binding = 2' in content
            assert 'binding = 3' in content

    def test_shader_uniforms(self):
        """Test that shaders have required uniforms"""
        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')

        # Field compute uniforms
        with open(os.path.join(shader_dir, 'field_compute.glsl'), 'r') as f:
            content = f.read()
            assert 'uniform int gridSize' in content
            assert 'uniform float time' in content
            assert 'uniform int samples' in content

        # Agent compute uniforms
        with open(os.path.join(shader_dir, 'agent_compute.glsl'), 'r') as f:
            content = f.read()
            assert 'uniform int gridSize' in content
            assert 'uniform int numAgents' in content

        # Vertex shader uniforms
        with open(os.path.join(shader_dir, 'vertex.glsl'), 'r') as f:
            content = f.read()
            assert 'uniform mat4 projection' in content
            assert 'uniform int gridSize' in content
            assert 'uniform vec2 windowSize' in content
            assert 'uniform float agentSize' in content


# These tests require GPU context - skipped in headless environments
class TestShaderManagerWithGPU:
    """Test shader compilation (requires OpenGL context)"""

    @pytest.fixture
    def mock_ctx(self):
        """Mock ModernGL context for testing"""
        # This would normally create a real context, but we skip it
        # in headless environments
        pytest.skip("Requires OpenGL context")

    def test_load_shader_source(self, mock_ctx):
        """Test loading shader source"""
        # Would test actual loading with context
        pytest.skip("Requires OpenGL context")

    def test_compile_compute_shader(self, mock_ctx):
        """Test compiling compute shader"""
        # Would test actual compilation with context
        pytest.skip("Requires OpenGL context")

    def test_compile_render_program(self, mock_ctx):
        """Test compiling render program"""
        # Would test actual compilation with context
        pytest.skip("Requires OpenGL context")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
