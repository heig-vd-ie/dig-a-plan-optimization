module ServerUtils

export get_server_config, ServerConfig

struct ServerConfig
    host::String
    port::Int
    base_url::String
end

function get_server_config(
    default_host = ENV["LOCAL_HOST"],
    default_port = ENV["SERVER_JL_PORT"],
    ;
    verbose = true,
)
    host = default_host

    port = if length(ARGS) >= 1
        parse(Int, ARGS[1])
    elseif haskey(ENV, "SERVER_JL_PORT")
        parse(Int, ENV["SERVER_JL_PORT"])
    else
        default_port
    end

    base_url = "http://$host:$port"

    if verbose
        println("Server configuration: $base_url")
    end

    return ServerConfig(host, port, base_url)
end

end # module
