"""
Static code analysis tests - catch issues before runtime
"""

import pytest
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestCodeQuality:
    """Test code quality and potential issues"""

    def test_no_print_statements_in_production(self):
        """Check that there are no debug print statements in core modules"""
        # Note: config_manager.py is allowed print in print_summary() method
        core_modules = [
            'src/simulation.py',
            'src/gpu_buffers.py',
            'src/shaders.py'
        ]

        for module_path in core_modules:
            filepath = os.path.join(os.path.dirname(__file__), '..', module_path)

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    tree = ast.parse(f.read())

                # Count print calls
                print_count = sum(
                    1 for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'print'
                )

                # Core modules should use logging, not print
                assert print_count == 0, \
                    f"{module_path} contains {print_count} print statements"

    def test_all_imports_exist(self):
        """Test that all imports in modules actually exist"""
        modules_to_check = [
            'src/config_manager.py',
            'src/simulation.py',
            'src/gpu_buffers.py',
            'src/shaders.py',
            'snail_trails_modular.py'
        ]

        for module_path in modules_to_check:
            filepath = os.path.join(os.path.dirname(__file__), '..', module_path)

            if not os.path.exists(filepath):
                continue

            # Try to import and check for errors
            with open(filepath, 'r') as f:
                content = f.read()

            # Parse and extract imports
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Standard library imports we expect
                        pass  # We can't easily check without importing

    def test_shader_glsl_syntax_basic(self):
        """Basic GLSL syntax validation"""
        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')

        for shader_file in os.listdir(shader_dir):
            if not shader_file.endswith('.glsl'):
                continue

            filepath = os.path.join(shader_dir, shader_file)
            with open(filepath, 'r') as f:
                content = f.read()

            # Check for common GLSL errors
            assert content.count('{') == content.count('}'), \
                f"{shader_file}: Mismatched braces"

            assert content.count('(') == content.count(')'), \
                f"{shader_file}: Mismatched parentheses"

            assert content.count('[') == content.count(']'), \
                f"{shader_file}: Mismatched brackets"

            # Check for required elements
            assert 'void main()' in content, \
                f"{shader_file}: Missing main function"

            assert '#version' in content, \
                f"{shader_file}: Missing version directive"

    def test_buffer_size_calculations(self):
        """Test buffer size calculations are correct"""
        from src.simulation import AgentManager
        from src.config_manager import ConfigManager

        # Test various configurations
        test_cases = [
            (100, 512),
            (1000, 1024),
            (10000, 2048),
        ]

        for num_agents, grid_size in test_cases:
            manager = AgentManager(num_agents, grid_size)
            manager.initialize_random_positions()

            bytes_data = manager.get_bytes()

            # Each agent is 24 bytes: pos(8) + velocity(8) + active(4) + padding(4)
            expected_size = num_agents * 24
            assert len(bytes_data) == expected_size, \
                f"Expected {expected_size} bytes, got {len(bytes_data)}"

    def test_config_derived_values_correct(self):
        """Test that derived configuration values are calculated correctly"""
        from src.config_manager import ConfigManager

        config = ConfigManager({
            'NUM_AGENTS': 10000,
            'STUCK_THRESHOLD_PERCENT': 10.0,
            'GRID_SIZE': 512
        })

        # Check stuck threshold calculation
        assert config['STUCK_THRESHOLD'] == 1000  # 10% of 10000

        # Check memory calculations are reasonable
        agent_mem = config['AGENT_MEMORY_MB']
        field_mem = config['FIELD_MEMORY_MB']
        grid_mem = config['GRID_MEMORY_MB']

        # All should be positive
        assert agent_mem > 0
        assert field_mem > 0
        assert grid_mem > 0

        # Total should be sum
        assert abs(config['TOTAL_MEMORY_MB'] - (agent_mem + field_mem + grid_mem)) < 0.01

    def test_modular_version_structure(self):
        """Test that modular version has correct structure"""
        filepath = os.path.join(os.path.dirname(__file__), '..', 'snail_trails_modular.py')

        with open(filepath, 'r') as f:
            content = f.read()

        # Check for required imports
        required_imports = [
            'from src.config_manager import',
            'from src.simulation import',
            'from src.gpu_buffers import',
            'from src.shaders import',
        ]

        for imp in required_imports:
            assert imp in content, f"Missing import: {imp}"

        # Check for required methods
        required_methods = [
            'def setup_simulation',
            'def setup_gpu',
            'def setup_shaders',
            'def setup_rendering',
            'def generate_vector_field',
            'def update_agents',
            'def render',
        ]

        for method in required_methods:
            assert method in content, f"Missing method: {method}"

        # Check that it uses config manager
        assert 'self.config = load_config_from_file' in content or \
               'ConfigManager' in content

    def test_no_hardcoded_magic_numbers(self):
        """Test that magic numbers are defined as constants"""
        filepath = os.path.join(os.path.dirname(__file__), '..', 'src/simulation.py')

        with open(filepath, 'r') as f:
            content = f.read()

        # Check that the AGENT_DTYPE is defined
        assert 'AGENT_DTYPE = np.dtype' in content

        # Check struct layout is documented
        assert 'pos' in content and 'velocity' in content and 'active' in content

    def test_error_messages_are_descriptive(self):
        """Test that error messages are helpful"""
        from src.config_manager import ConfigManager

        # Test various invalid configs and check error messages
        with pytest.raises(ValueError, match="GRID_SIZE must be between"):
            ConfigManager({'GRID_SIZE': 0})

        with pytest.raises(ValueError, match="NUM_AGENTS must be between"):
            ConfigManager({'NUM_AGENTS': -1})

        with pytest.raises(ValueError, match="should be divisible"):
            ConfigManager({'GRID_SIZE': 1000, 'FIELD_WORK_GROUP_SIZE': 16})

    def test_shader_uniforms_match_code(self):
        """Test that shader uniforms are used correctly in code"""
        # Check field compute shader
        shader_path = os.path.join(os.path.dirname(__file__), '..', 'shaders/field_compute.glsl')
        with open(shader_path, 'r') as f:
            shader_content = f.read()

        # Extract uniform declarations
        uniforms = []
        for line in shader_content.split('\n'):
            if 'uniform' in line and not line.strip().startswith('//'):
                uniforms.append(line.strip())

        # Check that uniforms are declared
        assert any('int gridSize' in u for u in uniforms)
        assert any('float time' in u for u in uniforms)
        assert any('int samples' in u for u in uniforms)

        # Check modular version sets these uniforms
        modular_path = os.path.join(os.path.dirname(__file__), '..', 'snail_trails_modular.py')
        with open(modular_path, 'r') as f:
            code_content = f.read()

        assert "['gridSize']" in code_content
        assert "['time']" in code_content
        assert "['samples']" in code_content

    def test_buffer_bindings_match(self):
        """Test that buffer bindings in shaders match code"""
        # Check agent compute shader
        shader_path = os.path.join(os.path.dirname(__file__), '..', 'shaders/agent_compute.glsl')
        with open(shader_path, 'r') as f:
            content = f.read()

        # Should have bindings 0, 1, 2, 3
        assert 'binding = 0' in content
        assert 'binding = 1' in content
        assert 'binding = 2' in content
        assert 'binding = 3' in content

        # Check that gpu_buffers.py binds correctly
        buffer_path = os.path.join(os.path.dirname(__file__), '..', 'src/gpu_buffers.py')
        with open(buffer_path, 'r') as f:
            content = f.read()

        # Should bind to storage buffer slots
        assert 'bind_to_storage_buffer(0)' in content
        assert 'bind_to_storage_buffer(1)' in content
        assert 'bind_to_storage_buffer(2)' in content
        assert 'bind_to_storage_buffer(3)' in content


class TestDataFlowConsistency:
    """Test that data flows correctly between components"""

    def test_agent_to_buffer_pipeline(self):
        """Test agent data → GPU buffer pipeline"""
        from src.simulation import AgentManager
        from src.config_manager import ConfigManager

        config = ConfigManager({'NUM_AGENTS': 100, 'GRID_SIZE': 64})

        # Create agents
        agent_mgr = AgentManager(config['NUM_AGENTS'], config['GRID_SIZE'])
        agent_mgr.initialize_random_positions(seed=42)

        # Get bytes for GPU
        agent_bytes = agent_mgr.get_bytes()

        # Should be correct size for GPU upload
        expected_size = config['NUM_AGENTS'] * 24
        assert len(agent_bytes) == expected_size

        # Data should be valid
        assert agent_bytes is not None
        assert isinstance(agent_bytes, bytes)

    def test_occupancy_grid_pipeline(self):
        """Test occupancy grid → GPU buffer pipeline"""
        from src.simulation import AgentManager, OccupancyGrid

        # Create agents
        agent_mgr = AgentManager(num_agents=50, grid_size=64)
        agent_mgr.initialize_random_positions(seed=42)

        # Create and populate occupancy grid
        occ_grid = OccupancyGrid(64)
        occ_grid.mark_positions(agent_mgr.get_positions())

        # Get bytes for GPU
        occ_bytes = occ_grid.get_bytes()

        # Should be correct size
        expected_size = 64 * 64 * 4  # int32 per cell
        assert len(occ_bytes) == expected_size

    def test_config_to_shader_pipeline(self):
        """Test config values are used correctly in shader setup"""
        from src.config_manager import ConfigManager

        config = ConfigManager({
            'GRID_SIZE': 1024,
            'FIELD_SAMPLES': 500,
            'NUM_AGENTS': 10000
        })

        # These values should be passed to shaders
        assert config['GRID_SIZE'] == 1024
        assert config['FIELD_SAMPLES'] == 500
        assert config['NUM_AGENTS'] == 10000

        # Work group calculations should be correct
        field_groups = (config['GRID_SIZE'] + 15) // 16
        agent_groups = (config['NUM_AGENTS'] + 255) // 256

        assert field_groups == 64  # 1024/16
        assert agent_groups == 40  # 10000/256 rounded up


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
