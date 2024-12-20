from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util import logging
from sphinx.directives.other import Include

logger = logging.getLogger(__name__)

class VersionListDirective(Directive):
    def run(self):
        env = self.state.document.settings.env
        versions = self.get_versions(env)
        
        # Create a bullet list
        bullet_list = nodes.bullet_list()
        
        # Add each version as a list item
        for version in versions:
            item = nodes.list_item()
            name = version['name']
            if name == env.app.config.version:
                name += " (Current)"
            para = nodes.paragraph()
            para += nodes.Text(name)
            item += para
            bullet_list += item
            
        return [bullet_list]
    
    def get_versions(self, env):
        versions_dir = Path(env.app.confdir) / "versions"
        versions = []
        
        if versions_dir.exists():
            version_dirs = [d for d in versions_dir.iterdir() if d.is_dir()]
            for ver_dir in version_dirs:
                versions.append({'name': ver_dir.name})
                
        # Sort versions in reverse order
        versions.sort(key=lambda x: x['name'], reverse=True)
        return versions

class VersionedIncludeDirective(Include):
    def run(self):
        env = self.state.document.settings.env
        current_version = env.app.config.version
        
        # Get the path argument and add version prefix
        original_path = self.arguments[0]
        self.arguments[0] = f"versions/{current_version}/{original_path}"
        
        # Debug logging
        logger.info(f"Including file: {self.arguments[0]}")
        logger.info(f"Current version: {current_version}")
        logger.info(f"Original path: {original_path}")
        logger.info(f"Full path: {Path(env.app.srcdir) / self.arguments[0]}")
        
        try:
            return super().run()
        except Exception as e:
            logger.error(f"Failed to include file: {self.arguments[0]}")
            logger.error(f"Error: {str(e)}")
            # Create a warning node instead of failing
            warning = nodes.warning()
            warning += nodes.paragraph(text=f"Could not include file {self.arguments[0]}: {str(e)}")
            return [warning]

def setup(app):
    app.add_directive('version-list', VersionListDirective)
    app.add_directive('versioned-include', VersionedIncludeDirective)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 