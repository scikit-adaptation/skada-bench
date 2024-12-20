from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging
from pathlib import Path
from myst_parser.main import MdParserConfig, parse_text

logger = logging.getLogger(__name__)

class VersionedInclude(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'parser': str,
        'class': str
    }

    def run(self):
        env = self.state.document.settings.env
        builder = env.app.builder
        
        # Get the base path of the file to include
        base_path = self.arguments[0]
        
        # Get current version from environment
        current_version = env.config.html_context.get('current_version', '')
        
        # Construct versioned path
        versioned_path = f"versions/{current_version}/{base_path}"
        
        logger.info(f"Loading versioned content from: {versioned_path}")
        
        # Create a paragraph node
        node = nodes.paragraph()
        node['classes'].append('versioned-content')
        
        # Add custom class if specified
        if 'class' in self.options:
            node['classes'].append(self.options['class'])
        
        # Store the original path as data attribute
        node['ids'].append(f'versioned-content')
        node.attributes['data-path'] = base_path
        
        try:
            with open(Path(env.srcdir) / versioned_path, 'r') as f:
                content = f.read()
                
            # Parse content based on parser option
            parser_name = self.options.get('parser', '')
            if parser_name == 'myst_parser.sphinx_':
                # Parse markdown content
                config = MdParserConfig()
                doc = parse_text(content, config=config)
                node.extend(doc.children)
            else:
                node += nodes.Text(content)
                
        except FileNotFoundError:
            logger.warning(f"Could not find versioned file: {versioned_path}")
            node += nodes.Text(f"Content not available for version {current_version}")
        except Exception as e:
            logger.error(f"Error processing file {versioned_path}: {str(e)}")
            node += nodes.Text(f"Error processing content for version {current_version}")
            
        return [node]

def setup(app):
    app.add_directive('versioned-include', VersionedInclude)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 