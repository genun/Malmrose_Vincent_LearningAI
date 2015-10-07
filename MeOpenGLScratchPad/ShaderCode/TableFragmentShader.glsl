#version 400

in vec2 daUV;
in mat3 where;
in vec3 fragPosition;
in vec3 normals;

uniform sampler2D textured;
uniform vec3 cameraPosition;
uniform vec3 light;

out vec3 FinalColor;


void main()
{
	vec3 min = vec3(0.0f, 0.0f, 0.0f);
	vec3 max = vec3(1.0f, 1.0f, 1.0f);

	//Get the Diffuse Lighting
	vec3 normalized = normalize(light - fragPosition);
	float diffuse = dot(normalized, normals);
	vec3 totalDiffuse = diffuse * vec3(1.0f, 1.0f, 1.0f);
	totalDiffuse = clamp(totalDiffuse, min, max);

	//Ambient Lighting
	float ambientColor = 0.15;
	vec3 ambient = vec3(ambientColor, ambientColor, ambientColor);
	vec3 lightingColor = clamp(totalDiffuse + ambient, min, max);
	
	vec3 postDiffuseLight = vec3(0.5f, 0.5f, 0.5f);//vec3(texture(textured, daUV));
	postDiffuseLight = lightingColor * postDiffuseLight;

	vec3 specularReflect = reflect(normalized * -1, normals);
	vec3 eyePosition = normalize(cameraPosition - fragPosition);
	float specularDotProduct = pow(dot(eyePosition, specularReflect), 3500 * 0.25);
	vec3 postSpecularLight = clamp(normalize(vec3(0.1f, 0.1f, 1.0f)) * specularDotProduct, min, max);
	
	FinalColor = vec3(texture(textured, daUV)) * postDiffuseLight + postSpecularLight;
	//FinalColor = vec3(texture(textured, daUV));
}