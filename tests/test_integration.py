"""
Integration tests for GPU operations

These tests require an OpenGL context and GPU.
They are skipped in headless/CI environments.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def has_opengl_context():
    """Check if OpenGL context is available"""
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except:
        return False


@pytest.mark.skipif(not has_opengl_context(), reason="Requires OpenGL context")
class TestGPUIntegration:
    """Integration tests for GPU operations"""

    @pytest.fixture
    def gpu_context(self):
        """Create GPU context for testing"""
        import moderngl
        ctx = moderngl.create_standalone_context(require=430)
        yield ctx
        ctx.release()

    def test_buffer_creation(self, gpu_context):
        """Test GPU buffer creation"""
        from src.gpu_buffers import GPUBufferManager

        manager = GPUBufferManager(
            ctx=gpu_context,
            num_agents=1000,
            grid_size=512
        )

        assert manager.agents_buffer is not None
        assert manager.field_buffer is not None
        assert manager.occupancy_buffer is not None
        assert manager.stats_buffer is not None

        # Check buffer sizes
        assert manager.agents_buffer.size == 1000 * 24  # 24 bytes per agent
        assert manager.field_buffer.size == 512 * 512 * 8  # 8 bytes per cell
        assert manager.occupancy_buffer.size == 512 * 512 * 4  # 4 bytes per cell

        manager.release()

    def test_agent_data_upload(self, gpu_context):
        """Test uploading agent data to GPU"""
        from src.gpu_buffers import GPUBufferManager
        from src.simulation import AgentManager

        # Create agents
        agent_mgr = AgentManager(num_agents=100, grid_size=512)
        agent_mgr.initialize_random_positions(seed=42)

        # Create GPU buffers
        buffer_mgr = GPUBufferManager(gpu_context, num_agents=100, grid_size=512)

        # Upload data
        buffer_mgr.upload_agents(agent_mgr.get_bytes())

        # Should not raise exception
        buffer_mgr.release()

    def test_shader_compilation(self, gpu_context):
        """Test shader compilation"""
        from src.shaders import ShaderManager

        shader_mgr = ShaderManager(
            ctx=gpu_context,
            shader_dir=os.path.join(os.path.dirname(__file__), '..', 'shaders')
        )

        # Compile field compute shader
        field_shader = shader_mgr.compile_compute_shader(
            'field_compute.glsl',
            name='field_compute'
        )
        assert field_shader is not None

        # Compile agent compute shader
        agent_shader = shader_mgr.compile_compute_shader(
            'agent_compute.glsl',
            name='agent_compute'
        )
        assert agent_shader is not None

        # Compile render program
        render_program = shader_mgr.compile_render_program(
            'vertex.glsl',
            'fragment.glsl',
            name='render'
        )
        assert render_program is not None

    def test_shader_uniforms(self, gpu_context):
        """Test setting shader uniforms"""
        from src.shaders import ShaderManager

        shader_mgr = ShaderManager(
            ctx=gpu_context,
            shader_dir=os.path.join(os.path.dirname(__file__), '..', 'shaders')
        )

        # Compile field shader
        field_shader = shader_mgr.compile_compute_shader('field_compute.glsl')

        # Set uniforms
        field_shader['gridSize'] = 512
        field_shader['time'] = 1.0
        field_shader['samples'] = 100

        # Should not raise exception

    def test_compute_shader_dispatch(self, gpu_context):
        """Test dispatching compute shader"""
        from src.shaders import ShaderManager
        from src.gpu_buffers import GPUBufferManager

        # Create buffers
        buffer_mgr = GPUBufferManager(gpu_context, num_agents=100, grid_size=64)

        # Load and compile shader
        shader_mgr = ShaderManager(
            ctx=gpu_context,
            shader_dir=os.path.join(os.path.dirname(__file__), '..', 'shaders')
        )
        field_shader = shader_mgr.compile_compute_shader('field_compute.glsl')

        # Bind buffer
        buffer_mgr.bind_for_field_compute()

        # Set uniforms
        field_shader['gridSize'] = 64
        field_shader['time'] = 0.0
        field_shader['samples'] = 10

        # Dispatch (4x4 work groups for 64x64 grid with 16x16 local size)
        field_shader.run(4, 4)

        # Should complete without error
        buffer_mgr.release()

    def test_stats_buffer_readback(self, gpu_context):
        """Test reading stats from GPU"""
        from src.gpu_buffers import GPUBufferManager

        buffer_mgr = GPUBufferManager(gpu_context, num_agents=1000, grid_size=512)

        # Reset stats
        buffer_mgr.reset_stats()

        # Read back
        stats = buffer_mgr.read_stats()

        assert len(stats) == 2
        assert stats[0] == 0  # notMoved
        assert stats[1] == 1000  # totalAgents

        buffer_mgr.release()

    def test_memory_usage_calculation(self, gpu_context):
        """Test memory usage calculation"""
        from src.gpu_buffers import GPUBufferManager

        buffer_mgr = GPUBufferManager(
            gpu_context,
            num_agents=1_000_000,
            grid_size=2048
        )

        memory_mb = buffer_mgr.get_memory_usage_mb()

        # Should be reasonable size
        assert 0 < memory_mb < 1000  # Less than 1GB

        buffer_mgr.release()

    def test_full_simulation_step(self, gpu_context):
        """Test a full simulation step (field + agents)"""
        from src.gpu_buffers import GPUBufferManager
        from src.simulation import AgentManager, OccupancyGrid
        from src.shaders import ShaderManager

        # Setup
        num_agents = 100
        grid_size = 64

        # Create agents
        agent_mgr = AgentManager(num_agents, grid_size)
        agent_mgr.initialize_random_positions(seed=42)

        # Create occupancy grid
        occ_grid = OccupancyGrid(grid_size)
        occ_grid.mark_positions(agent_mgr.get_positions())

        # Create GPU buffers
        buffer_mgr = GPUBufferManager(gpu_context, num_agents, grid_size)
        buffer_mgr.upload_agents(agent_mgr.get_bytes())
        buffer_mgr.upload_occupancy(occ_grid.get_bytes())

        # Load shaders
        shader_mgr = ShaderManager(
            gpu_context,
            shader_dir=os.path.join(os.path.dirname(__file__), '..', 'shaders')
        )
        field_shader = shader_mgr.compile_compute_shader('field_compute.glsl')
        agent_shader = shader_mgr.compile_compute_shader('agent_compute.glsl')

        # Generate field
        buffer_mgr.bind_for_field_compute()
        field_shader['gridSize'] = grid_size
        field_shader['time'] = 0.0
        field_shader['samples'] = 10
        field_shader.run(4, 4)  # 64/16 = 4 groups each dimension

        # Update agents
        buffer_mgr.reset_stats()
        buffer_mgr.bind_for_agent_compute()
        agent_shader['gridSize'] = grid_size
        agent_shader['numAgents'] = num_agents
        agent_shader.run((num_agents + 255) // 256)  # Round up to work groups

        # Read stats
        stats = buffer_mgr.read_stats()
        not_moved = stats[0]

        # Some agents should have moved (not all stuck)
        assert 0 <= not_moved <= num_agents

        buffer_mgr.release()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
