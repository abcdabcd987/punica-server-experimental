use std::collections::hash_map::{Entry, HashMap};

use uuid::Uuid;

pub struct PagedKvTracker {
    block_len: u32,
    free_blocks: u32,
    requests: HashMap<Uuid, RequestContext>,
}

struct RequestContext {
    num_pages: u32,
    last_page_len: u32,
}

impl PagedKvTracker {
    pub fn new(kvpool_capacity: u32, block_len: u32) -> Self {
        Self {
            block_len,
            free_blocks: kvpool_capacity,
            requests: HashMap::new(),
        }
    }

    pub fn num_free_blocks(&self) -> u32 {
        self.free_blocks
    }

    pub fn calc_init_blocks(&self, seqlen: u32) -> u32 {
        (seqlen + self.block_len - 1) / self.block_len
    }

    pub fn init(&mut self, request_id: Uuid, seqlen: u32) -> bool {
        let entry = match self.requests.entry(request_id) {
            Entry::Occupied(_) => {
                panic!("Request ID already exists {}", request_id)
            }
            Entry::Vacant(entry) => entry,
        };
        let num_pages = (seqlen + self.block_len - 1) / self.block_len;
        if num_pages > self.free_blocks {
            false
        } else {
            self.free_blocks -= num_pages;
            let last_page_len = (seqlen - 1) % self.block_len + 1;
            entry.insert(RequestContext { num_pages, last_page_len });
            true
        }
    }

    pub fn append_token(&mut self, request_id: &Uuid) -> bool {
        let ctx =
            self.requests.get_mut(request_id).expect("Request ID not found");
        if ctx.last_page_len < self.block_len {
            ctx.last_page_len += 1;
            true
        } else if self.free_blocks > 0 {
            self.free_blocks -= 1;
            ctx.num_pages += 1;
            ctx.last_page_len = 1;
            true
        } else {
            false
        }
    }

    pub fn release(&mut self, request_id: Uuid) {
        let ctx =
            self.requests.remove(&request_id).expect("Request ID not found");
        self.free_blocks += ctx.num_pages;
    }

    pub fn request_len(&self, request_id: &Uuid) -> u32 {
        let ctx = self.requests.get(request_id).expect("Request ID not found");
        (ctx.num_pages - 1) * self.block_len + ctx.last_page_len
    }
}
